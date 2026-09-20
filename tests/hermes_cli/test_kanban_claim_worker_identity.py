"""Every dispatch path must leave a claimed task with a usable worker identity.

``release_stale_claims`` only extends (rather than reclaims) a TTL-expired claim when
``host_local and worker_pid and _worker_alive(...)``. Regression for the case where a
claim opened outside the dispatcher's spawn path left ``worker_pid``/``worker_started_at``
NULL forever, so the live-worker guard could never be true and a healthy worker was
reclaimed out from under itself.
"""

import json
import os
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    connection = kbc.connect(db)
    yield connection
    connection.close()


def _row(conn, task_id):
    return conn.execute(
        "SELECT status, claim_lock, worker_pid, worker_started_at FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()


def test_claim_task_records_live_worker_identity(conn):
    """The pull lane (``hermes kanban claim`` / ``claim_task``) is a dispatch path too."""
    tid = kb.create_task(conn, title="t", assignee="default")
    kb.claim_task(conn, tid)

    row = _row(conn, tid)
    assert row["status"] == "running"
    assert row["worker_pid"] == os.getpid()
    assert row["worker_started_at"] is not None
    # host_local must be computable from the lock this claim minted.
    assert row["claim_lock"].startswith(kb._host_prefix())


def test_claim_review_task_records_live_worker_identity(conn):
    tid = kb.create_task(conn, title="t", assignee="default")
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
    conn.commit()

    kb.claim_review_task(conn, tid)

    row = _row(conn, tid)
    assert row["worker_pid"] == os.getpid()
    assert row["worker_started_at"] is not None


def test_explicit_worker_pid_overrides_the_claiming_process(conn):
    """A pull-lane caller that claims on behalf of a longer-lived process names it."""
    tid = kb.create_task(conn, title="t", assignee="default")
    kb.claim_task(conn, tid, worker_pid=os.getppid())

    assert _row(conn, tid)["worker_pid"] == os.getppid()


def _expire_claim(conn, task_id):
    """Push the claim's lease into the past (TTL is floored at 1s, so set it directly)."""
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 60, task_id))
    conn.commit()


def test_expired_claim_with_live_worker_is_extended_not_reclaimed(conn):
    """The acceptance criterion: a demonstrably-alive worker keeps its claim.

    The worker is a separate live process (pid 1, always alive and not the sweeper),
    matching the dispatcher shape where the sweeper and the worker are different
    processes. A row naming the sweeper itself is the claim-without-spawn case and
    must still be reclaimed, so it cannot stand in for a worker here.
    """
    tid = kb.create_task(conn, title="t", assignee="default")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, 1)  # the dispatcher's spawn record
    _expire_claim(conn, tid)

    assert kb.release_stale_claims(conn) == 0
    assert _row(conn, tid)["status"] == "running"
    kinds = [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]
    assert "claim_extended" in kinds
    assert "reclaimed" not in kinds


def test_expired_claim_naming_only_the_sweeper_still_reclaims(conn):
    """A claim that never spawned a worker must not be kept alive by its own claimer.

    The claiming process stamps its pid so the sweeper can see a real worker; that
    must not become a self-referential liveness proof that extends the claim forever
    and stops the failure breaker from ever tripping.
    """
    tid = kb.create_task(conn, title="t", assignee="default")
    kb.claim_task(conn, tid)  # pid == this process, no spawn
    _expire_claim(conn, tid)

    assert kb.release_stale_claims(conn) == 1
    assert _row(conn, tid)["status"] == "ready"


def test_sweeping_a_self_claimed_row_never_signals_this_process(conn):
    """A claim naming the sweeping process must never be SIGTERMed.

    Every claim stamps the claiming process's pid, so an in-process claimer that later
    runs the sweeper would otherwise kill itself — the reclaim path signals whatever
    ``worker_pid`` names, and only the spawn record distinguishes a real worker from
    the claimer.
    """
    tid = kb.create_task(conn, title="t", assignee="default")
    kb.claim_task(conn, tid)
    _expire_claim(conn, tid)

    signalled = []
    assert kb.release_stale_claims(
        conn, signal_fn=lambda pid, sig: signalled.append((pid, sig))) == 1
    assert signalled == [], f"sweeper signalled itself: {signalled}"


def test_max_runtime_enforcement_ignores_a_run_that_never_spawned(conn):
    """The runtime limiter signals directly, so it must skip claimer-only rows too."""
    tid = kb.create_task(conn, title="t", assignee="default")
    kb.claim_task(conn, tid)
    conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
    conn.execute(
        "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
        (int(time.time()) - 3600, tid))
    conn.commit()

    signalled = []
    timed_out = kbd.enforce_max_runtime(
        conn, signal_fn=lambda pid, sig: signalled.append((pid, sig)))
    assert signalled == [], f"runtime enforcer signalled its own claimer: {signalled}"
    assert timed_out == []


def test_reclaim_event_reports_the_real_host_local_and_names_the_failed_guard(conn):
    """Diagnostic: the reclaim event must say WHICH guard component failed.

    ``host_local`` was previously clobbered by the termination report (which reports
    False whenever there is no pid to signal), so every reclaim claimed the lock was
    foreign even when it was minted on this very host.
    """
    tid = kb.create_task(conn, title="t", assignee="default")
    kb.claim_task(conn, tid)
    # Spawned, then the worker died: the guard must fail on liveness, not host or pid.
    kbd._set_worker_pid(conn, tid, os.getpid())
    conn.execute("UPDATE tasks SET worker_pid = 2147483646 WHERE id = ?", (tid,))
    _expire_claim(conn, tid)

    assert kb.release_stale_claims(conn) == 1

    payload = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'reclaimed'", (tid,)
    ).fetchone()["payload"]
    data = json.loads(payload)
    assert data["host_local"] is True, "lock was minted on this host"
    assert data["guard_failed"] == ["worker_alive"]


def test_heartbeat_adopts_a_worker_identity_the_claim_path_could_not_record(conn):
    """Legacy/foreign-claimed rows converge on a real pid via the worker's own heartbeat."""
    tid = kb.create_task(conn, title="t", assignee="default")
    kb.claim_task(conn, tid)
    conn.execute(
        "UPDATE tasks SET worker_pid = NULL, worker_started_at = NULL WHERE id = ?", (tid,))
    conn.commit()

    assert kbd.heartbeat_worker(conn, tid) is True

    row = _row(conn, tid)
    assert row["worker_pid"] == os.getpid()
    assert row["worker_started_at"] is not None


def test_heartbeat_does_not_steal_a_live_workers_identity(conn):
    """An ephemeral CLI heartbeat must not repoint a row at itself while the worker lives."""
    tid = kb.create_task(conn, title="t", assignee="default")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getppid())

    kbd.heartbeat_worker(conn, tid)

    assert _row(conn, tid)["worker_pid"] == os.getppid()

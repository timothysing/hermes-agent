"""A dispatcher-claimed task must keep its lease for as long as its worker lives.

The lease has two independent keepers and both had a hole:

* ``heartbeat_claim`` authorised on a freshly computed ``f"{host}:{os.getpid()}"``
  instead of the claim the worker actually holds, so a worker spawned by the
  dispatcher could never extend the lease it was given — every heartbeat was a
  silent no-op and the task was reclaimed at the TTL while the board showed a
  healthy stream of heartbeats.
* ``release_stale_claims`` only spares a TTL-expired claim when
  ``host_local and worker_pid and worker_alive``; dispatch paths that left
  ``worker_pid`` NULL disabled that safety net entirely.

These tests drive the lifecycle the way the dispatcher does (claim in one
process identity, heartbeat under another) rather than the in-process shape,
which is the only shape the old code got right.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (mirrors test_kanban_db.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", raising=False)
    kb.init_db()
    return home


def _dispatcher_lock() -> str:
    """A host-local lock belonging to some OTHER process than this one."""
    host = kb._claimer_id().split(":", 1)[0]
    return f"{host}:1111"


def _expires(conn, task_id):
    return conn.execute(
        "SELECT claim_expires FROM tasks WHERE id = ?", (task_id,)).fetchone()["claim_expires"]


def _status(conn, task_id):
    return conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"]


def _claimed_task(conn, *, lock=None, ttl=60):
    task_id = kb.create_task(conn, title="lease", assignee="a")
    kb.claim_task(conn, task_id, claimer=lock or _dispatcher_lock(), ttl_seconds=ttl)
    return task_id


# ---------------------------------------------------------------------------
# heartbeat_claim: the worker extends the claim it was handed
# ---------------------------------------------------------------------------


def test_worker_in_another_pid_extends_the_dispatchers_claim(kanban_home, monkeypatch):
    """THE regression: claim held by 'host:1111', heartbeat from this (different) pid.

    The worker presents the lock it was spawned with via
    ``HERMES_KANBAN_CLAIM_LOCK``. Old code recomputed ``host:<own pid>``, which
    can never match a dispatcher-minted lock, so this returned False and the
    lease never moved.
    """
    lock = _dispatcher_lock()
    with kbc.connect() as conn:
        task_id = _claimed_task(conn, lock=lock)
        before = _expires(conn, task_id)
        assert lock != kb._claimer_id(), "test must heartbeat from a different pid"

        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", lock)
        assert kb.heartbeat_claim(conn, task_id, ttl_seconds=3600) is True
        assert _expires(conn, task_id) > before


def test_worker_extends_via_the_rows_stored_lock_when_no_lock_is_presented(kanban_home):
    """No env lock, no explicit claimer: authorise on the row's own host-local lock.

    This is the plain ``kanban_heartbeat`` call inside a worker whose env was not
    pinned; recomputing ``host:<pid>`` silently refused it.
    """
    with kbc.connect() as conn:
        task_id = _claimed_task(conn)
        before = _expires(conn, task_id)

        assert kb.heartbeat_claim(conn, task_id, ttl_seconds=3600) is True
        assert _expires(conn, task_id) > before


def test_foreign_claim_lock_is_refused_and_the_lease_does_not_move(kanban_home):
    """Authorisation is narrowed, not removed: someone else's claim stays theirs."""
    with kbc.connect() as conn:
        task_id = _claimed_task(conn, lock="some-other-host:4242")
        before = _expires(conn, task_id)

        assert kb.heartbeat_claim(conn, task_id, ttl_seconds=3600, claimer="wrong:9") is False
        assert kb.heartbeat_claim(conn, task_id, ttl_seconds=3600) is False
        assert _expires(conn, task_id) == before


def test_heartbeat_does_not_extend_a_task_that_is_not_running(kanban_home, monkeypatch):
    """A lease belongs to a running claim; a finished/parked task keeps its own."""
    lock = _dispatcher_lock()
    with kbc.connect() as conn:
        task_id = _claimed_task(conn, lock=lock)
        before = _expires(conn, task_id)
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        conn.commit()

        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", lock)
        assert kb.heartbeat_claim(conn, task_id, ttl_seconds=3600) is False
        assert _expires(conn, task_id) == before


# ---------------------------------------------------------------------------
# release_stale_claims: the live-worker safety net
# ---------------------------------------------------------------------------


def _expire_lease(conn, task_id):
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 60, task_id))
    conn.commit()


def test_expired_lease_with_a_live_spawned_worker_is_extended_not_reclaimed(kanban_home):
    """Host-local claim + a recorded, live worker pid: the claim is extended.

    pid 1 stands in for the spawned worker: always alive and never this process
    (a row naming the sweeper itself is the claim-without-worker case below).
    """
    with kbc.connect() as conn:
        task_id = _claimed_task(conn)
        kbd._set_worker_pid(conn, task_id, 1)
        _expire_lease(conn, task_id)
        before = _expires(conn, task_id)

        assert kb.release_stale_claims(conn) == 0
        assert _status(conn, task_id) == "running"
        assert _expires(conn, task_id) > before
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))]
        assert "claim_extended" in kinds and "reclaimed" not in kinds


def test_expired_lease_with_a_dead_worker_is_still_reclaimed(kanban_home):
    """No regression in the recovery path a genuinely dead worker depends on."""
    with kbc.connect() as conn:
        task_id = _claimed_task(conn)
        kbd._set_worker_pid(conn, task_id, os.getpid())  # spawn record...
        conn.execute(  # ...for a pid that is gone by sweep time
            "UPDATE tasks SET worker_pid = 2147483646 WHERE id = ?", (task_id,))
        _expire_lease(conn, task_id)

        assert kb.release_stale_claims(conn) == 1
        assert _status(conn, task_id) == "ready"


def test_expired_lease_with_no_worker_pid_is_reclaimed(kanban_home):
    """A claim that never recorded a worker cannot be spared — nothing proves life."""
    with kbc.connect() as conn:
        task_id = _claimed_task(conn)
        conn.execute(
            "UPDATE tasks SET worker_pid = NULL, worker_started_at = NULL WHERE id = ?",
            (task_id,))
        _expire_lease(conn, task_id)

        assert kb.release_stale_claims(conn) == 1
        assert _status(conn, task_id) == "ready"


# ---------------------------------------------------------------------------
# The whole point: a long build survives many TTL windows on heartbeats alone
# ---------------------------------------------------------------------------


class _Clock:
    """``time`` stand-in whose ``time()`` we advance by hand."""

    def __init__(self, start: float):
        self._now = start

    def __getattr__(self, name):  # strftime / fromisoformat / sleep pass through
        return getattr(time, name)

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_long_run_heartbeating_once_a_minute_survives_many_ttl_windows(
        kanban_home, monkeypatch):
    """90 simulated minutes at a 5-minute TTL: worker heartbeats, sweeper sweeps.

    The task has no live spawned worker to fall back on, so ONLY the heartbeat
    keeps the lease alive — exactly the long-build shape that was reclaimed at
    15 minutes.
    """
    lock = _dispatcher_lock()
    clock = _Clock(time.time())
    monkeypatch.setattr(kb, "time", clock)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", lock)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", "300")

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="long build", assignee="a")
        kb.claim_task(conn, task_id, claimer=lock)
        conn.execute(  # no spawned worker: the heartbeat is the only keeper
            "UPDATE tasks SET worker_pid = NULL, worker_started_at = NULL WHERE id = ?",
            (task_id,))
        conn.commit()

        for minute in range(1, 91):
            clock.advance(60)
            assert kb.heartbeat_claim(conn, task_id) is True, f"heartbeat lost at {minute}m"
            assert kb.release_stale_claims(conn) == 0, f"reclaimed at {minute}m"
            assert _status(conn, task_id) == "running", f"not running at {minute}m"

        assert _expires(conn, task_id) > clock.time()

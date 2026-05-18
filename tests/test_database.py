import pytest
import os
import tempfile
from unittest.mock import patch


@pytest.fixture(autouse=True)
def temp_db(monkeypatch):
    """Use a temporary SQLite file for each test."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("DB_PATH", tmp.name)
    monkeypatch.setenv("AGENT_MODE", "custom")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CLAY_API_KEY", "test-key")
    monkeypatch.setenv("VIBE_API_KEY", "test-key")
    yield tmp.name
    os.unlink(tmp.name)


def test_init_db_creates_tables(temp_db):
    import importlib
    import config as cfg
    importlib.reload(cfg)
    import database as db
    importlib.reload(db)
    db.init_db()
    with db.get_connection() as conn:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
    assert "contacted" in tables
    assert "approved_messages" in tables
    assert "review_queue" in tables
    assert "signal_history" in tables
    assert "run_stats" in tables


def test_mark_and_check_contacted(temp_db):
    import importlib
    import config as cfg
    importlib.reload(cfg)
    import database as db
    importlib.reload(db)
    db.init_db()

    assert db.is_contacted("apollo-001") is False
    db.mark_contacted("apollo-001", "Jane Smith", "Sunrise Medical", "run-001")
    assert db.is_contacted("apollo-001") is True


def test_mark_contacted_idempotent(temp_db):
    import importlib
    import config as cfg
    importlib.reload(cfg)
    import database as db
    importlib.reload(db)
    db.init_db()

    db.mark_contacted("apollo-002", "John Doe", "HealthCo", "run-001")
    db.mark_contacted("apollo-002", "John Doe", "HealthCo", "run-002")
    with db.get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM contacted WHERE apollo_id = ?", ("apollo-002",)
        ).fetchone()[0]
    assert count == 1


def test_signal_history_update(temp_db):
    import importlib
    import config as cfg
    importlib.reload(cfg)
    import database as db
    importlib.reload(db)
    db.init_db()

    db.update_signal_history("open_rcm_jobs", approved=True)
    db.update_signal_history("open_rcm_jobs", approved=True)
    db.update_signal_history("open_rcm_jobs", approved=False)

    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT attempts, approvals FROM signal_history WHERE signal_type = ?",
            ("open_rcm_jobs",)
        ).fetchone()
    assert row["attempts"] == 3
    assert row["approvals"] == 2


def test_get_signal_weights_neutral_below_threshold(temp_db):
    import importlib
    import config as cfg
    importlib.reload(cfg)
    import database as db
    importlib.reload(db)
    db.init_db()

    db.update_signal_history("work_history", approved=True)
    db.update_signal_history("work_history", approved=False)
    weights = db.get_signal_weights()
    assert weights["work_history"] == 0.5


def test_get_signal_weights_above_threshold(temp_db):
    import importlib
    import config as cfg
    importlib.reload(cfg)
    import database as db
    importlib.reload(db)
    db.init_db()

    for _ in range(8):
        db.update_signal_history("open_rcm_jobs", approved=True)
    for _ in range(2):
        db.update_signal_history("open_rcm_jobs", approved=False)
    weights = db.get_signal_weights()
    assert abs(weights["open_rcm_jobs"] - 0.8) < 0.01

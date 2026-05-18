import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Generator, List, Optional
from loguru import logger
from config import settings


def init_db() -> None:
    """Create all tables if they don't exist. Called once at pipeline startup."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS contacted (
                apollo_id     TEXT PRIMARY KEY,
                name          TEXT,
                company       TEXT,
                processed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                run_id        TEXT
            );

            CREATE TABLE IF NOT EXISTS approved_messages (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                apollo_id      TEXT,
                name           TEXT,
                company        TEXT,
                email          TEXT,
                signal_type    TEXT,
                signal_content TEXT,
                message        TEXT,
                specificity    REAL,
                relevance      REAL,
                naturalness    REAL,
                total_score    REAL,
                processed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                run_id         TEXT
            );

            CREATE TABLE IF NOT EXISTS review_queue (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                apollo_id    TEXT,
                name         TEXT,
                company      TEXT,
                signal_type  TEXT,
                message      TEXT,
                total_score  REAL,
                feedback     TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                run_id       TEXT
            );

            CREATE TABLE IF NOT EXISTS signal_history (
                signal_type  TEXT PRIMARY KEY,
                attempts     INTEGER DEFAULT 0,
                approvals    INTEGER DEFAULT 0,
                last_updated TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS searched_companies (
                domain        TEXT NOT NULL,
                searched_date TEXT NOT NULL,
                PRIMARY KEY (domain, searched_date)
            );

            CREATE TABLE IF NOT EXISTS run_stats (
                run_id           TEXT PRIMARY KEY,
                started_at       TIMESTAMP,
                completed_at     TIMESTAMP,
                mode             TEXT,
                prospects_found  INTEGER DEFAULT 0,
                enriched         INTEGER DEFAULT 0,
                approved         INTEGER DEFAULT 0,
                review_needed    INTEGER DEFAULT 0,
                skipped          INTEGER DEFAULT 0,
                errors           INTEGER DEFAULT 0,
                duration_seconds REAL DEFAULT 0
            );
        """)
    logger.info("Database initialized")


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def is_contacted(apollo_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM contacted WHERE apollo_id = ?", (apollo_id,)
        ).fetchone()
        return row is not None


def mark_contacted(apollo_id: str, name: str, company: str, run_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO contacted (apollo_id, name, company, run_id) VALUES (?, ?, ?, ?)",
            (apollo_id, name, company, run_id),
        )
    logger.debug(f"Marked {name} ({apollo_id}) as contacted")


def save_approved(record: Dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO approved_messages
            (apollo_id, name, company, email, signal_type, signal_content, message,
             specificity, relevance, naturalness, total_score, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["apollo_id"], record["name"], record["company"], record["email"],
                record["signal_type"], record["signal_content"], record["message"],
                record["specificity"], record["relevance"], record["naturalness"],
                record["total_score"], record["run_id"],
            ),
        )


def save_review(record: Dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO review_queue
            (apollo_id, name, company, signal_type, message, total_score, feedback, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["apollo_id"], record["name"], record["company"],
                record["signal_type"], record["message"], record["total_score"],
                record["feedback"], record["run_id"],
            ),
        )


def update_signal_history(signal_type: str, approved: bool) -> None:
    approval_inc = 1 if approved else 0
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO signal_history (signal_type, attempts, approvals, last_updated)
            VALUES (?, 1, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(signal_type) DO UPDATE SET
                attempts     = attempts + 1,
                approvals    = approvals + ?,
                last_updated = CURRENT_TIMESTAMP
            """,
            (signal_type, approval_inc, approval_inc),
        )


def get_signal_weights() -> Dict[str, float]:
    """Load per-signal approval rates for adaptive ranking. Neutral (0.5) until 5+ attempts."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT signal_type,
                   CASE WHEN attempts >= 5
                        THEN CAST(approvals AS REAL) / attempts
                        ELSE 0.5
                   END AS approval_rate
            FROM signal_history
            """
        ).fetchall()
    return {row["signal_type"]: row["approval_rate"] for row in rows}


def has_searched_company(domain: str) -> bool:
    """Return True if Clay find-at-company was already called for this domain today."""
    today = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM searched_companies WHERE domain = ? AND searched_date = ?",
            (domain.lower().strip(), today),
        ).fetchone()
    return row is not None


def mark_company_searched(domain: str) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO searched_companies (domain, searched_date) VALUES (?, ?)",
            (domain.lower().strip(), today),
        )


def get_approved_messages(limit: int = 100) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM approved_messages ORDER BY processed_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_review_queue(limit: int = 100) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM review_queue ORDER BY processed_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_run_history(limit: int = 20) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM run_stats ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_dashboard_stats() -> Dict[str, Any]:
    with get_connection() as conn:
        approved = conn.execute("SELECT COUNT(*) FROM approved_messages").fetchone()[0]
        review = conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0]
        contacted = conn.execute("SELECT COUNT(*) FROM contacted").fetchone()[0]
        runs = conn.execute("SELECT COUNT(*) FROM run_stats").fetchone()[0]
    return {"approved": approved, "review_needed": review,
            "contacted": contacted, "total_runs": runs}


def get_approved_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM approved_messages WHERE id = ?", (record_id,)
        ).fetchone()
    return dict(row) if row else None


def save_run_stats(stats: Dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO run_stats
            (run_id, started_at, completed_at, mode, prospects_found, enriched,
             approved, review_needed, skipped, errors, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stats["run_id"], stats["started_at"], stats["completed_at"],
                stats["mode"], stats["prospects_found"], stats["enriched"],
                stats["approved"], stats["review_needed"], stats["skipped"],
                stats["errors"], stats["duration_seconds"],
            ),
        )

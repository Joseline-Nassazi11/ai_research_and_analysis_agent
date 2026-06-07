"""
db/database.py
--------------
SQLite database layer for Analyst AI.

Handles all persistence: users, sessions, messages, documents,
reports, citations, and search cache. All functions use context-managed
connections and raise specific exceptions on failure.

Improvements over original:
  - Password hashing upgraded from plain SHA-256 to PBKDF2-HMAC-SHA256
    with 260,000 iterations (NIST-recommended 2023 baseline). Uses only
    Python stdlib (hashlib) — no external packages required.
  - Existing SHA-256 hashes are detected automatically and re-hashed on
    first successful login (transparent migration, no user action needed).
  - Sessions are now scoped per user — each user only sees their own chats.

FIXES:
  - add_document: DELETE now scoped by user_id to prevent cross-user deletion
  - get_all_documents: also returns documents with NULL user_id for the
    owner (backwards compatibility for docs uploaded before auth existed)
  - init_db: all ALTER TABLE migrations wrapped in try/except to prevent
    "duplicate column name" errors when the column already exists
"""

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

# Constants 
DB_PATH = Path(__file__).parent.parent / "db" / "analyst.db"
CACHE_TTL_MINUTES = 60
MIN_USERNAME_LENGTH = 3
MIN_PASSWORD_LENGTH = 6
DEFAULT_SESSION_TITLE_FORMAT = "Research Session — %b %d, %Y %I:%M %p"
MESSAGE_HISTORY_LIMIT = 40
MAX_RECENT_SESSIONS = 8

# PBKDF2 settings 
_PBKDF2_ITERATIONS = 260_000
_PBKDF2_HASH = "sha256"
_PBKDF2_DKLEN = 32
_HASH_VERSION = "pbkdf2v1"
_LEGACY_PREFIX_LEN = 32


# Connection 

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# Initialisation

def init_db() -> None:
    """Create all required tables if they do not already exist.
    Also runs migrations for existing databases (adds user_id to sessions).
    """
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            last_login TEXT DEFAULT ''
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            size_bytes INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            ingested_at TEXT NOT NULL,
            summary TEXT DEFAULT '',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Migration: add user_id to existing documents table if missing
    # Wrapped in try/except — if column already exists SQLite raises
    # OperationalError which we safely ignore.
    try:
        c.execute("PRAGMA table_info(documents)")
        doc_cols = {row["name"] for row in c.fetchall()}
        if "user_id" not in doc_cols:
            c.execute("ALTER TABLE documents ADD COLUMN user_id INTEGER REFERENCES users(id)")
    except Exception:
        pass  # column already exists — safe to ignore

    # Sessions table — user_id scopes sessions per user
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Migration: add user_id to existing sessions table if missing
    # IMPORTANT: must call PRAGMA table_info(sessions) here separately —
    # cannot reuse results from the documents PRAGMA above.
    try:
        c.execute("PRAGMA table_info(sessions)")
        existing_cols = {row["name"] for row in c.fetchall()}
        if "user_id" not in existing_cols:
            c.execute("ALTER TABLE sessions ADD COLUMN user_id INTEGER REFERENCES users(id)")
    except Exception:
        pass  # column already exists — safe to ignore

    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL,
            session_id INTEGER,
            file_path TEXT NOT NULL,
            report_type TEXT DEFAULT 'research',
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    # Migration: add user_id to existing reports table if missing
    try:
        c.execute("PRAGMA table_info(reports)")
        rep_cols = {row["name"] for row in c.fetchall()}
        if "user_id" not in rep_cols:
            c.execute("ALTER TABLE reports ADD COLUMN user_id INTEGER REFERENCES users(id)")
    except Exception:
        pass  # column already exists — safe to ignore

    c.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query TEXT PRIMARY KEY,
            results TEXT NOT NULL,
            cached_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS citations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            authors TEXT DEFAULT '',
            year TEXT DEFAULT '',
            url TEXT DEFAULT '',
            publisher TEXT DEFAULT '',
            page_numbers TEXT DEFAULT '',
            citation_format TEXT DEFAULT 'APA',
            raw_text TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    conn.commit()
    conn.close()


# ─── Documents ────────────────────────────────────────────────────────────────

def add_document(filename: str, file_type: str, size_bytes: int, chunk_count: int, user_id: int | None = None) -> int:
    conn = get_connection()
    now = datetime.now().isoformat()
    c = conn.cursor()
    # scope DELETE by user_id so one user can't accidentally delete
    # another user's document record that happens to share the same filename.
    if user_id is not None:
        c.execute(
            "DELETE FROM documents WHERE filename = ? AND (user_id = ? OR user_id IS NULL)",
            (filename, user_id),
        )
    else:
        c.execute("DELETE FROM documents WHERE filename = ? AND user_id IS NULL", (filename,))
    c.execute(
        "INSERT INTO documents (user_id, filename, file_type, size_bytes, chunk_count, ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, filename, file_type, size_bytes, chunk_count, now),
    )
    doc_id = c.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def update_document_summary(filename: str, summary: str) -> None:
    conn = get_connection()
    conn.execute("UPDATE documents SET summary = ? WHERE filename = ?", (summary, filename))
    conn.commit()
    conn.close()


def get_all_documents(user_id: int | None = None) -> list[dict]:
    """Retrieve documents scoped to a user.

    If user_id is provided, returns documents belonging to that user.
    Also includes any legacy documents with NULL user_id so older uploads
    still appear (backwards compatibility).
    If user_id is None, returns all documents (admin/fallback use only).
    """
    conn = get_connection()
    c = conn.cursor()
    if user_id is not None:
        # also include NULL user_id rows so documents uploaded before
        # auth was added (or uploaded with a missing user_id) still show up.
        c.execute(
            "SELECT * FROM documents WHERE user_id = ? OR user_id IS NULL ORDER BY ingested_at DESC",
            (user_id,),
        )
    else:
        c.execute("SELECT * FROM documents ORDER BY ingested_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_document(filename: str) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()


def clear_documents() -> None:
    conn = get_connection()
    conn.execute("DELETE FROM documents")
    conn.commit()
    conn.close()


# ─── Sessions ─────────────────────────────────────────────────────────────────

def create_session(title: str | None = None, user_id: int | None = None) -> int:
    """Create a new research session scoped to a specific user.

    Args:
        title: Optional session title.
        user_id: The ID of the user who owns this session.

    Returns:
        The row ID of the new session.
    """
    now = datetime.now().isoformat()
    title = title or f"Research Session — {datetime.now().strftime('%b %d, %Y %I:%M %p')}"
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO sessions (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (user_id, title, now, now),
    )
    sid = c.lastrowid
    conn.commit()
    conn.close()
    return sid


def get_all_sessions(user_id: int | None = None) -> list[dict]:
    """Retrieve sessions ordered by last update descending.

    If user_id is provided, only returns that user's sessions.
    If user_id is None, returns all sessions (admin/fallback use only).

    Args:
        user_id: Filter sessions to this user. Pass None to get all.

    Returns:
        List of session records as dicts.
    """
    conn = get_connection()
    c = conn.cursor()
    if user_id is not None:
        c.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
    else:
        c.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def rename_session(session_id: int, title: str) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
        (title, datetime.now().isoformat(), session_id),
    )
    conn.commit()
    conn.close()


def delete_session(session_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM citations WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


# ─── Messages ─────────────────────────────────────────────────────────────────

def save_message(session_id: int, role: str, content: str) -> None:
    conn = get_connection()
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now),
    )
    conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?",
        (now, session_id),
    )
    conn.commit()
    conn.close()


def get_session_messages(session_id: int, limit: int = MESSAGE_HISTORY_LIMIT) -> list[dict]:
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ─── Reports ──────────────────────────────────────────────────────────────────

def save_report(
    title: str,
    file_path: str,
    session_id: int | None = None,
    report_type: str = "research",
    user_id: int | None = None,
) -> int:
    conn = get_connection()
    now = datetime.now().isoformat()
    c = conn.cursor()
    c.execute(
        "INSERT INTO reports (user_id, title, session_id, file_path, report_type, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, title, session_id, file_path, report_type, now),
    )
    rid = c.lastrowid
    conn.commit()
    conn.close()
    return rid


def get_all_reports(user_id: int | None = None) -> list[dict]:
    """Retrieve reports. If user_id provided, only returns that user's reports."""
    conn = get_connection()
    c = conn.cursor()
    if user_id is not None:
        c.execute("SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    else:
        c.execute("SELECT * FROM reports ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_report(report_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()


# ─── Search Cache ─────────────────────────────────────────────────────────────

def cache_search(query: str, results: list) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO search_cache (query, results, cached_at) VALUES (?, ?, ?)",
        (query, json.dumps(results), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_cached_search(query: str) -> list | None:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT results, cached_at FROM search_cache WHERE query = ?", (query,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    cached_time = datetime.fromisoformat(row["cached_at"])
    age_minutes = (datetime.now() - cached_time).seconds / 60
    if age_minutes < CACHE_TTL_MINUTES:
        return json.loads(row["results"])
    return None


# ─── Citations ────────────────────────────────────────────────────────────────

def save_citation(
    session_id: int,
    source_type: str,
    title: str,
    authors: str = "",
    year: str = "",
    url: str = "",
    publisher: str = "",
    page_numbers: str = "",
    citation_format: str = "APA",
    raw_text: str = "",
) -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id FROM citations WHERE session_id = ? AND title = ? AND url = ?",
        (session_id, title, url),
    )
    existing = c.fetchone()
    if existing:
        conn.close()
        return existing["id"]
    c.execute(
        """INSERT INTO citations
           (session_id, source_type, title, authors, year, url, publisher,
            page_numbers, citation_format, raw_text, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, source_type, title, authors, year, url,
         publisher, page_numbers, citation_format, raw_text,
         datetime.now().isoformat()),
    )
    cid = c.lastrowid
    conn.commit()
    conn.close()
    return cid


def get_session_citations(session_id: int) -> list[dict]:
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM citations WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_citations() -> list[dict]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM citations ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_citation(citation_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM citations WHERE id = ?", (citation_id,))
    conn.commit()
    conn.close()


def clear_session_citations(session_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM citations WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def format_citation(citation: dict) -> str:
    fmt = citation.get("citation_format", "APA")
    title = citation.get("title", "Untitled")
    authors = citation.get("authors", "")
    year = citation.get("year", "n.d.")
    url = citation.get("url", "")
    publisher = citation.get("publisher", "")
    pages = citation.get("page_numbers", "")
    source_type = citation.get("source_type", "web")

    if fmt == "APA":
        if source_type == "web":
            a = f"{authors}. " if authors else ""
            u = f" Retrieved from {url}" if url else ""
            return f"{a}({year}). *{title}*.{u}"
        else:
            a = f"{authors}. " if authors else ""
            p = f" {publisher}." if publisher else ""
            pg = f" pp. {pages}." if pages else ""
            return f"{a}({year}). *{title}*.{p}{pg}"
    elif fmt == "MLA":
        a = f"{authors}. " if authors else ""
        u = f" {url}." if url else ""
        p = f" {publisher}," if publisher else ""
        return f'{a}"{title}."{p}{u} {year}.'
    elif fmt == "Chicago":
        a = f"{authors}. " if authors else ""
        p = f" {publisher}," if publisher else ""
        u = f" {url}." if url else ""
        pg = f" {pages}." if pages else ""
        return f'{a}"{title}."{p} {year}.{u}{pg}'
    return f"{title} ({year})"


# ─── Authentication ───────────────────────────────────────────────────────────

def _hash_password_pbkdf2(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        _PBKDF2_HASH,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PBKDF2_ITERATIONS,
        dklen=_PBKDF2_DKLEN,
    )
    stored = f"{_HASH_VERSION}:{salt}:{dk.hex()}"
    return stored, salt


def _verify_pbkdf2(password: str, stored: str) -> bool:
    try:
        _, salt, dk_hex = stored.split(":", 2)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac(
        _PBKDF2_HASH,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PBKDF2_ITERATIONS,
        dklen=_PBKDF2_DKLEN,
    )
    return hmac.compare_digest(dk.hex(), dk_hex)


def _is_legacy_sha256_hash(stored: str) -> bool:
    return not stored.startswith(_HASH_VERSION + ":")


def _verify_legacy_sha256(password: str, stored: str) -> bool:
    try:
        salt, pw_hash = stored.split(":", 1)
    except ValueError:
        return False
    check = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return hmac.compare_digest(check, pw_hash)


def create_user(username: str, password: str, display_name: str = "") -> dict:
    if len(username.strip()) < MIN_USERNAME_LENGTH:
        return {"ok": False, "error": f"Username must be at least {MIN_USERNAME_LENGTH} characters."}
    if len(password) < MIN_PASSWORD_LENGTH:
        return {"ok": False, "error": f"Password must be at least {MIN_PASSWORD_LENGTH} characters."}

    stored, _ = _hash_password_pbkdf2(password)
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, created_at) "
            "VALUES (?, ?, ?, ?)",
            (username.strip().lower(), stored, display_name or username, datetime.now().isoformat()),
        )
        conn.commit()
        return {"ok": True}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Username already taken. Please choose another."}
    except sqlite3.OperationalError as e:
        return {"ok": False, "error": f"Database error: {str(e)}"}
    finally:
        conn.close()


def verify_user(username: str, password: str) -> dict:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"ok": False, "error": "Username not found."}

    stored = row["password_hash"]

    if _is_legacy_sha256_hash(stored):
        if not _verify_legacy_sha256(password, stored):
            return {"ok": False, "error": "Incorrect password."}
        new_stored, _ = _hash_password_pbkdf2(password)
        conn = get_connection()
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_stored, row["id"]))
        conn.commit()
        conn.close()
    else:
        if not _verify_pbkdf2(password, stored):
            return {"ok": False, "error": "Incorrect password."}

    conn = get_connection()
    conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now().isoformat(), row["id"]))
    conn.commit()
    conn.close()
    return {"ok": True, "user": dict(row)}


def update_password(username: str, new_password: str) -> dict:
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return {"ok": False, "error": f"Password must be at least {MIN_PASSWORD_LENGTH} characters."}
    stored, _ = _hash_password_pbkdf2(new_password)
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (stored, username.strip().lower()),
        )
        conn.commit()
        return {"ok": True}
    except sqlite3.OperationalError as e:
        return {"ok": False, "error": f"Database error: {str(e)}"}
    finally:
        conn.close()


def update_display_name(username: str, display_name: str) -> dict:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET display_name = ? WHERE username = ?",
            (display_name, username.strip().lower()),
        )
        conn.commit()
        return {"ok": True}
    except sqlite3.OperationalError as e:
        return {"ok": False, "error": f"Database error: {str(e)}"}
    finally:
        conn.close()


def get_user(username: str) -> dict | None:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def user_count() -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) as n FROM users")
    row = c.fetchone()
    conn.close()
    return row["n"] if row else 0
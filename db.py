"""db.py

Small psycopg2 connection helper shared by ingest.py and match.py.

Configure via environment variables (falls back to sane local defaults):
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
"""

from __future__ import annotations

import os
from contextlib import contextmanager


def get_connection():
    try:
        import psycopg2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ValueError("Song recognition needs psycopg2. Run: py -3 -m pip install psycopg2-binary") from error
    try:
        return psycopg2.connect(
            host=os.environ.get("PGHOST", "localhost"),
            port=os.environ.get("PGPORT", "5432"),
            dbname=os.environ.get("PGDATABASE", "shazam_poc"),
            user=os.environ.get("PGUSER", "postgres"),
            password=os.environ.get("PGPASSWORD", ""),
        )
    except psycopg2.OperationalError as error:
        raise ValueError(
            "Can't reach the song-recognition database. Make sure PostgreSQL is running, "
            "the 'shazam_poc' database exists (see schema.sql), and PGUSER/PGPASSWORD are "
            "set if needed."
        ) from error


@contextmanager
def cursor():
    """with cursor() as cur: ... — commits on success, rolls back on error."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

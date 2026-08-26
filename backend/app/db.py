"""Database engine, session handling, and declarative base."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_settings = get_settings()

engine = create_engine(
    _settings.database_url,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


#: Columns added after a database was first created.
#:
#: `create_all` creates missing tables but never alters an existing one, so a
#: developer database from before a column existed comes back up silently
#: missing it and fails at query time instead of at startup. Each entry is
#: idempotent; this list goes away when Alembic takes over.
_ADDITIVE_COLUMNS = (
    "ALTER TABLE actions ADD COLUMN IF NOT EXISTS scheduled_for TIMESTAMPTZ",
    "CREATE INDEX IF NOT EXISTS ix_actions_scheduled_for ON actions (scheduled_for)",
)


def init_db() -> None:
    """Create tables. Alembic takes over once the schema stabilises."""
    from sqlalchemy import text

    from app import models  # noqa: F401  (import registers the mappers)

    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        for statement in _ADDITIVE_COLUMNS:
            conn.execute(text(statement))

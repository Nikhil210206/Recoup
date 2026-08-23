"""Fixtures for integration tests.

These run against a **separate database** (`recoup_test`), created here and torn
down here. That separation is not tidiness. These fixtures call `drop_all()`, and
pointing that at the development database destroys real captured data -- which is
exactly what happened on day 0. See INCIDENTS.md.

Requires Postgres (`make db`).
"""

from __future__ import annotations

import os

import pytest

# This must run before anything imports app.db, which builds its engine at import
# time from settings. Environment variables take precedence over .env in
# pydantic-settings, so assigning here redirects the whole session.
_DEV_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://recoup:recoup@localhost:5434/recoup"
)
TEST_DB_NAME = "recoup_test"
_TEST_URL = _DEV_URL.rsplit("/", 1)[0] + f"/{TEST_DB_NAME}"

os.environ["DATABASE_URL"] = _TEST_URL
os.environ.setdefault("RAZORPAY_WEBHOOK_SECRET", "whsec_test_example")


def _ensure_test_database() -> None:
    """CREATE DATABASE recoup_test if absent. Connects to `postgres` to do it."""
    import psycopg

    admin_dsn = (
        _DEV_URL.replace("postgresql+psycopg://", "postgresql://").rsplit("/", 1)[0] + "/postgres"
    )
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB_NAME,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')


@pytest.fixture(scope="session")
def _engine():
    _ensure_test_database()

    from app import models  # noqa: F401  (registers mappers)
    from app.db import Base, engine

    # Hard stop: never let the destructive fixtures touch a non-test database.
    assert engine.url.database == TEST_DB_NAME, (
        f"refusing to run drop_all against database {engine.url.database!r}; "
        f"expected {TEST_DB_NAME!r}"
    )

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(_engine):
    from app import models
    from app.db import SessionLocal

    session = SessionLocal()
    yield session
    # Truncate between tests rather than rolling back: the webhook handler
    # commits internally, so a transaction-scoped rollback would not undo it.
    for table in reversed(models.Base.metadata.sorted_tables):
        session.execute(table.delete())
    session.commit()
    session.close()


@pytest.fixture
def client(_engine):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c

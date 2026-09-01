"""Things that only break on a managed host.

Every test here guards a failure that does not reproduce locally. On a laptop
the database URL already names a driver, the API process is never cold, and
nobody ever sets a live key by accident. On Render all three change at once, and
each one fails in a way that looks like something else.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import make_url

from app.config import Settings

BACKEND = Path(__file__).resolve().parents[1]


class TestDatabaseUrl:
    """Managed Postgres does not spell the URL the way SQLAlchemy needs."""

    @pytest.mark.parametrize(
        "given",
        [
            "postgres://u:p@host:5432/db",  # Heroku's legacy form, still emitted
            "postgresql://u:p@host:5432/db",  # what Render hands out
        ],
    )
    def test_managed_host_urls_are_routed_to_psycopg3(self, given):
        """A bare `postgresql://` resolves to psycopg2, which is NOT installed --
        requirements.txt pins `psycopg[binary]`, which is psycopg 3.

        The failure mode is a ModuleNotFoundError raised at import time, before
        logging exists, so on a managed host it presents as a service that will
        not boot for no stated reason. Normalising is the fix; this test is what
        stops the normaliser being removed as apparently redundant.
        """
        url = Settings(database_url=given).database_url
        assert url.startswith("postgresql+psycopg://")
        assert make_url(url).get_dialect().driver == "psycopg"

    def test_an_explicit_driver_is_left_alone(self):
        """The local default already names psycopg. Rewriting it would be a
        second place for the connection string to be wrong."""
        given = "postgresql+psycopg://recoup:recoup@localhost:5434/recoup"
        assert Settings(database_url=given).database_url == given

    def test_credentials_survive_the_rewrite(self):
        """Rewriting a URL scheme by string surgery is exactly the kind of edit
        that quietly drops a password containing a colon or an @."""
        given = "postgres://user:p%40ss:word@host:5432/db?sslmode=require"
        url = make_url(Settings(database_url=given).database_url)
        assert url.username == "user"
        assert url.password == "p@ss:word"
        assert url.host == "host"
        assert url.port == 5432
        assert url.database == "db"
        assert url.query["sslmode"] == "require"


class TestLiveKeyRefusal:
    """Rule 4 of this project: test mode only. It had no test until now."""

    def test_a_live_key_is_refused(self):
        with pytest.raises(ValueError, match="test-mode only"):
            Settings(razorpay_key_id="rzp_live_ABCDEF123456")

    def test_a_test_key_is_accepted(self):
        s = Settings(razorpay_key_id="rzp_test_ABCDEF123456")
        assert s.razorpay_key_id == "rzp_test_ABCDEF123456"


class TestColdStartImports:
    """The API must not drag the ML stack into a process that serves webhooks.

    `app.main` used to import pandas, numpy, scikit-learn, scipy, pyarrow and
    joblib transitively through api/actions.py -- about 378MB of site-packages,
    roughly a second of import time. On a host that sleeps when idle, that cost
    is paid on the cold start a Razorpay webhook wakes up. Razorpay times out,
    retries, and the idempotency guard rejects the retry, so the case is never
    classified. That is the Day 4 incident, re-entering through the deployment
    instead of through the handler.

    The imports are therefore inside the functions that need them. Nothing about
    that looks deliberate to a reader tidying imports, which is why this test
    exists and says so.
    """

    FORBIDDEN = ("pandas", "sklearn", "scipy", "pyarrow", "joblib")

    def test_importing_the_app_does_not_load_the_ml_stack(self):
        # A subprocess, because by the time this test runs the parent has
        # imported all of these already and sys.modules would always pass.
        code = (
            "import sys, json; import app.main; "
            f"print(json.dumps([m for m in {self.FORBIDDEN!r} if m in sys.modules]))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=BACKEND,
            capture_output=True,
            text=True,
            check=True,
        )
        loaded = eval(out.stdout.strip())  # noqa: S307 - our own json, one line
        assert loaded == [], (
            f"importing app.main pulled {loaded} into the webhook process. "
            "Something re-hoisted an allocator import to module scope; put it "
            "back inside the function that uses it."
        )

    def test_the_allocator_still_works_when_imported_lazily(self):
        """Deferring an import is only safe if the deferred thing still runs.

        A NameError inside `_build_allocator` would not surface until the first
        allocation on the deployed service -- past every test that only checks
        the app starts.
        """
        from app.api.actions import _build_allocator

        allocator = _build_allocator(budget=10)
        assert allocator.budget_policy.max_total_contacts == 10

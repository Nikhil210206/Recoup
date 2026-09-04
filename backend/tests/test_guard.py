"""The token on the automation endpoints and the privileged parameters.

Written the way the risk is shaped rather than the way the code is: each test
names the thing that goes wrong if the guard is absent.
"""

from __future__ import annotations

import pytest

from app.api.guard import HEADER
from app.config import Settings

pytestmark = pytest.mark.integration

TOKEN = "test-token-not-a-real-secret"


@pytest.fixture
def secured(monkeypatch):
    """A process configured the way the deployed one is."""

    def _settings():
        return Settings(tasks_token=TOKEN, app_env="production")

    for module in ("app.api.guard",):
        monkeypatch.setattr(f"{module}.get_settings", _settings)
    return TOKEN


class TestTheSchedulerEndpointsAreClosed:
    def test_execute_due_is_refused_without_a_token(self, client, secured):
        assert client.post("/tasks/execute-due").status_code == 401

    def test_classify_pending_is_refused_without_a_token(self, client, secured):
        assert client.post("/tasks/classify-pending").status_code == 401

    def test_a_wrong_token_is_refused(self, client, secured):
        res = client.post("/tasks/execute-due", headers={HEADER: "nearly-" + TOKEN})
        assert res.status_code == 401

    def test_the_right_token_is_accepted(self, client, secured):
        res = client.post("/tasks/execute-due", headers={HEADER: TOKEN})
        assert res.status_code == 200, res.text

    def test_allocate_is_closed_too(self, client, secured):
        """`/actions/allocate` is automation wearing a different prefix."""
        assert client.post("/actions/allocate").status_code == 401


class TestThePrivilegedParametersAreClosed:
    def test_force_cannot_be_set_anonymously(self, client, secured):
        """`force=true` writes `case.schedule_overridden` to the ledger as a
        human decision. An anonymous caller must never be able to author that:
        the ledger is the artifact this project's central claim rests on."""
        assert client.post("/tasks/execute-due?force=true").status_code == 401

    def test_live_in_a_request_body_is_caught(self, client, secured, db_session):
        """`approve` takes `live` in the BODY. A guard that only inspected the
        query string would wave this through and it reaches Razorpay."""
        res = client.post(
            "/actions/does-not-matter/approve",
            json={"approved_by": "attacker", "execute_now": True, "live": True},
        )
        # 401 before the 404 for the unknown action: the guard runs first, so a
        # caller cannot probe for valid action ids either.
        assert res.status_code == 401


class TestTheConsoleStillWorksForAJudge:
    """Locking the doors is easy. Locking them without locking out the person
    the deployment exists for is the actual requirement."""

    def test_reads_are_open(self, client, secured):
        for path in ("/health", "/api/overview", "/api/rules", "/actions/pending"):
            assert client.get(path).status_code == 200, path

    def test_the_demonstration_is_open(self, client, secured):
        res = client.post("/api/demo/simulate", json={"error_reason": "card_expired"})
        assert res.status_code == 200, res.text

    def test_approving_without_live_is_open(self, client, secured):
        """The console sends live=false. That path must stay clickable."""
        res = client.post(
            "/actions/no-such-action/approve",
            json={"approved_by": "console", "execute_now": True, "live": False},
        )
        # 404 rather than 401: it got past the guard and failed on the id.
        assert res.status_code == 404


class TestFailureIsClosedInProduction:
    def test_production_without_a_token_refuses_rather_than_serving_open(
        self, client, monkeypatch
    ):
        """A missing secret is a deployment mistake. Continuing to serve the
        endpoint unauthenticated is the wrong response to it."""
        monkeypatch.setattr(
            "app.api.guard.get_settings",
            lambda: Settings(tasks_token="", app_env="production"),
        )
        assert client.post("/tasks/execute-due").status_code == 503

    def test_development_without_a_token_stays_usable(self, client, monkeypatch):
        monkeypatch.setattr(
            "app.api.guard.get_settings",
            lambda: Settings(tasks_token="", app_env="dev"),
        )
        assert client.post("/tasks/execute-due").status_code == 200

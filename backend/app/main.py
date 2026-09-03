"""Recoup -- the revenue recovery control plane.

Every recovery agent asks "can I recover this?" Recoup asks "should I?"
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import actions, dashboard, demo, health, tasks, webhooks
from app.db import init_db

STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Recoup",
    description="The revenue recovery control plane.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(webhooks.router)
app.include_router(tasks.router)
app.include_router(actions.router)
app.include_router(dashboard.router)
app.include_router(demo.router)


# The console is a static page over the /api read models. Mounted last so it can
# never shadow an API route, and mounted at /static rather than / so that adding
# an endpoint later cannot silently start returning HTML.
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def console() -> HTMLResponse:
    """Serve the console with its assets versioned by content.

    Without this the browser keeps a stylesheet from a previous deploy and the
    page renders new markup against old rules -- which looks like a layout bug
    and is not one. The stamp is the newest asset mtime, so it changes exactly
    when something it refers to changes.
    """
    stamp = max(int(path.stat().st_mtime) for path in STATIC.glob("*.*"))
    html = (STATIC / "index.html").read_text()
    html = html.replace("/static/styles.css", f"/static/styles.css?v={stamp}")
    html = html.replace("/static/app.js", f"/static/app.js?v={stamp}")
    return HTMLResponse(html)

"""Recoup -- the revenue recovery control plane.

Every recovery agent asks "can I recover this?" Recoup asks "should I?"
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import actions, health, tasks, webhooks
from app.db import init_db


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

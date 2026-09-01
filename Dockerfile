# Recoup -- single service: FastAPI serves the API and the console.
#
# The build context is the REPOSITORY ROOT, not backend/. This is deliberate and
# load-bearing: `config.REPO_ROOT` resolves to `backend/app/config.py`'s
# parents[2], and the dashboard reads `docs/evidence/*.json` relative to it. A
# build context of backend/ produces an image that starts cleanly and then
# answers 503 on the console's evaluation panel, because the evidence files are
# simply not there. Copy them in.

FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Dependencies first, so a code change does not reinstall ~380MB of wheels.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application code, then the evidence the console reads at runtime.
COPY backend/app ./backend/app
COPY backend/pytest.ini ./backend/pytest.ini
COPY docs/evidence ./docs/evidence

# Run unprivileged. Nothing here writes to disk: the ledger is in Postgres and
# the evidence files are read-only.
RUN useradd --create-home --uid 10001 recoup \
    && chown -R recoup:recoup /srv
USER recoup

WORKDIR /srv/backend

# Render injects $PORT and it is not optional -- a service that binds a fixed
# port is marked unhealthy and never receives traffic. 8010 is the local default
# and matches `make api`.
ENV PORT=8010
EXPOSE 8010

# No --reload. One worker: the allocator's fitted estimator is cached per
# process, so additional workers would each refit it, and the free tier has
# neither the memory nor the traffic to want more than one.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8010} --workers 1"]

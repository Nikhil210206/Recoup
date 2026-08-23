.PHONY: help venv install db db-down api test test-all lint fmt clean

PY := backend/.venv/bin/python
PIP := backend/.venv/bin/pip

help:
	@echo "Recoup -- the revenue recovery control plane"
	@echo ""
	@echo "  make install    create venv and install backend deps"
	@echo "  make db         start Postgres (docker compose)"
	@echo "  make api        run the FastAPI server on :8000"
	@echo "  make test       unit tests (no database needed)"
	@echo "  make test-all   unit + integration tests (needs make db)"
	@echo "  make lint       ruff check"
	@echo "  make clean      stop db, remove venv and caches"

venv:
	@test -d backend/.venv || python3 -m venv backend/.venv

install: venv
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r backend/requirements.txt

db:
	docker compose up -d db
	@echo "waiting for postgres..."
	@until docker compose exec -T db pg_isready -U recoup -d recoup >/dev/null 2>&1; do sleep 1; done
	@echo "postgres ready on localhost:$${RECOUP_DB_PORT:-5434}"

db-down:
	docker compose down

api:
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

test:
	cd backend && .venv/bin/pytest

test-all:
	cd backend && .venv/bin/pytest -m ""

lint:
	cd backend && .venv/bin/ruff check app tests

fmt:
	cd backend && .venv/bin/ruff format app tests

clean:
	docker compose down -v
	rm -rf backend/.venv backend/.pytest_cache backend/.ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

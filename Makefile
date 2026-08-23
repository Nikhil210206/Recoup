.PHONY: help venv install db db-down api data data-worlds assumptions arms arms-worlds classifier-eval ollama model test test-all lint fmt clean

PY := backend/.venv/bin/python
PIP := backend/.venv/bin/pip

help:
	@echo "Recoup -- the revenue recovery control plane"
	@echo ""
	@echo "  make install    create venv and install backend deps"
	@echo "  make db         start Postgres (docker compose)"
	@echo "  make api        run the FastAPI server on :8000"
	@echo "  make data       generate the synthetic dataset (seed 42, base world)"
	@echo "  make data-worlds generate all three world parameterisations"
	@echo "  make assumptions regenerate data/ASSUMPTIONS.md from code"
	@echo "  make arms       compare recovery arms on the test split"
	@echo "  make arms-worlds compare arms across all three worlds"
	@echo "  make ollama     start the local model runtime (free, no API key)"
	@echo "  make classifier-eval  measure the LLM tail on held-out error codes"
	@echo "  make model      train + calibrate the uplift model, run the lever study"
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

data:
	cd backend && .venv/bin/python -m app.simulation.cli --seed $(or $(SEED),42) --world base

data-worlds:
	cd backend && for w in base pessimistic optimistic; do \
		.venv/bin/python -m app.simulation.cli --seed $(or $(SEED),42) --world $$w; \
	done

assumptions:
	cd backend && .venv/bin/python -m app.simulation.docgen

arms:
	cd backend && .venv/bin/python -m app.simulation.arms --world $(or $(WORLD),base) --split $(or $(SPLIT),test)

arms-worlds:
	cd backend && for w in pessimistic base optimistic; do \
		.venv/bin/python -m app.simulation.arms --world $$w --split test; echo; \
	done

ollama:
	@pgrep -x ollama >/dev/null || (ollama serve > /tmp/ollama.log 2>&1 &)
	@sleep 2 && ollama list

classifier-eval:
	cd backend && .venv/bin/python -m app.services.classifier_eval \
		--models $(or $(MODELS),ollama:llama3.2:3b,ollama:qwen2.5:7b)

model:
	cd backend && .venv/bin/python -m app.model.train --world $(or $(WORLD),base) --seed $(or $(SEED),42) --save

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

# gdrag v2 — Developer Makefile
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help setup setup-venv setup-docker setup-db api mcp test test-unit \
        test-file lint format type-check clean clean-docker logs ps

PYTHON   ?= python3
VENV     ?= .venv
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
UVICORN  := $(VENV)/bin/uvicorn

# Default target
help:
	@echo "gdrag v2 — available targets:"
	@echo ""
	@echo "  Setup"
	@echo "    make setup          First-time setup: venv + deps + Docker + DB"
	@echo "    make setup-venv     Create virtual environment and install deps"
	@echo "    make setup-docker   Start Docker services (PostgreSQL + Qdrant)"
	@echo "    make setup-db       Run database migrations"
	@echo ""
	@echo "  Run"
	@echo "    make api            Start FastAPI server (http://localhost:8000)"
	@echo "    make mcp            Start MCP v2 server"
	@echo ""
	@echo "  Test"
	@echo "    make test           Run full test suite"
	@echo "    make test-unit      Run unit tests only (no DB/network)"
	@echo "    make test-file FILE=tests/test_foo.py   Run a specific test file"
	@echo "    make coverage       Run tests with HTML coverage report"
	@echo ""
	@echo "  Quality"
	@echo "    make lint           Run ruff linter"
	@echo "    make format         Auto-format with ruff"
	@echo "    make type-check     Run mypy type checker"
	@echo ""
	@echo "  Docker"
	@echo "    make logs           Show docker-compose logs"
	@echo "    make ps             Show running containers"
	@echo "    make clean-docker   Stop and remove containers + volumes"
	@echo ""
	@echo "  Misc"
	@echo "    make clean          Remove __pycache__, .pytest_cache, build artifacts"

# ─── Setup ────────────────────────────────────────────────────────────────────

setup: setup-venv setup-docker setup-db
	@echo ""
	@echo "✓ gdrag v2 ready. Run 'make api' to start the server."

setup-venv:
	@echo "→ Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt
	@echo "✓ Virtual environment ready at $(VENV)/"

setup-docker:
	@echo "→ Starting Docker services..."
	docker compose up -d postgres qdrant
	@echo "→ Waiting for PostgreSQL to be ready..."
	@sleep 3
	@until docker compose exec postgres pg_isready -U gdrag -q 2>/dev/null; do \
		echo "  Waiting for PostgreSQL..."; sleep 2; \
	done
	@echo "✓ Docker services running."

setup-db:
	@echo "→ Running database migrations..."
	@if [ -f migrations/001_session_memory.sql ]; then \
		docker compose exec -T postgres psql -U gdrag -d gdrag < migrations/001_session_memory.sql; \
		echo "  ✓ Migration 001 applied"; \
	fi
	@if [ -f migrations/002_enhanced_knowledge.sql ]; then \
		docker compose exec -T postgres psql -U gdrag -d gdrag < migrations/002_enhanced_knowledge.sql; \
		echo "  ✓ Migration 002 applied"; \
	fi
	@echo "✓ Migrations complete."

# ─── Run ─────────────────────────────────────────────────────────────────────

api:
	@echo "→ Starting FastAPI server on http://localhost:8000 ..."
	@echo "   Docs: http://localhost:8000/docs"
	PYTHONPATH=. $(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000 --reload

mcp:
	@echo "→ Starting MCP v2 server..."
	PYTHONPATH=. $(PYTHON) -m src.mcp.v2.server

# ─── Test ─────────────────────────────────────────────────────────────────────

test:
	PYTHONPATH=. $(PYTEST) tests/ -v

test-unit:
	PYTHONPATH=. $(PYTEST) tests/ -v -m "not integration and not slow"

test-file:
	@test -n "$(FILE)" || (echo "Usage: make test-file FILE=tests/test_foo.py" && exit 1)
	PYTHONPATH=. $(PYTEST) $(FILE) -v

coverage:
	PYTHONPATH=. $(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "→ Coverage report: htmlcov/index.html"

# ─── Quality ──────────────────────────────────────────────────────────────────

lint:
	$(VENV)/bin/ruff check src/ tests/

format:
	$(VENV)/bin/ruff check --fix src/ tests/
	$(VENV)/bin/ruff format src/ tests/

type-check:
	$(VENV)/bin/mypy src/ --ignore-missing-imports

# ─── Docker ───────────────────────────────────────────────────────────────────

logs:
	docker compose logs -f

ps:
	docker compose ps

clean-docker:
	docker compose down -v

# Full stack with optional services
up-full:
	docker compose --profile full up -d

up-graph:
	docker compose --profile graph up -d

up-cache:
	docker compose --profile cache up -d

# ─── Misc ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Cleaned."

# ============================================================================
# XAU/USD Scalping Bot — developer convenience targets
# ============================================================================
.DEFAULT_GOAL := help
PY ?= python3
PIP ?= $(PY) -m pip

.PHONY: help env install backend frontend test docker-up docker-down docker-build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "Created .env (edit to add API keys)")

install: env ## Install backend + frontend dependencies
	$(PIP) install -r requirements.txt
	cd frontend && npm install

backend: env ## Run the FastAPI backend (port 8000)
	cd backend && uvicorn main:app --reload --port 8000

frontend: ## Run the Vite dev server (port 5173)
	cd frontend && npm run dev

test: ## Run the backend test suite (offline synthetic data)
	XAU_DATA_PROVIDER=synthetic $(PY) -m pytest backend/tests -q

docker-build: ## Build docker images
	docker compose build

docker-up: env ## Start the full stack with docker compose
	docker compose up -d --build
	@echo "Dashboard: http://localhost:5173  |  API: http://localhost:8000/docs"

docker-down: ## Stop the docker stack
	docker compose down

clean: ## Remove caches and the local SQLite DB
	rm -rf backend/__pycache__ backend/tests/__pycache__ .pytest_cache
	rm -f backend/xau_bot.db backend/xau_bot.db-wal backend/xau_bot.db-shm

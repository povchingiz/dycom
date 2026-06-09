VENV = .venv312
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.PHONY: help setup run stop clean

help:
	@echo "FaceSim Pipeline"
	@echo ""
	@echo "  make setup   — install everything (run once on first deploy)"
	@echo "  make run     — start the web demo server"
	@echo "  make stop    — stop the server"
	@echo "  make clean   — remove sessions and temp files"

setup:
	@echo "==> Creating Python 3.12 environment..."
	python3.12 -m venv $(VENV)
	@echo "==> Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "==> Created .env from example — EDIT IT: set DEMO_PASSWORD before running"; \
	else \
		echo "==> .env already exists, skipping"; \
	fi
	@echo ""
	@echo "Done. Next steps:"
	@echo "  1. Edit .env — set DEMO_PASSWORD and SEGMENTATION_DEVICE"
	@echo "  2. Run: make run"

run:
	@if [ ! -f .env ]; then echo "ERROR: .env not found. Run: make setup"; exit 1; fi
	@grep -q "change_this_password" .env && echo "ERROR: Change DEMO_PASSWORD in .env first" && exit 1 || true
	@echo "==> Starting FaceSim server at http://0.0.0.0:8000 ..."
	cd server && ../$(VENV)/bin/uvicorn main:app --host 0.0.0.0 --port 8000

stop:
	@pkill -f "uvicorn main:app" && echo "Server stopped" || echo "Server not running"

clean:
	@echo "==> Cleaning sessions..."
	rm -rf server/sessions/*
	@echo "Done"

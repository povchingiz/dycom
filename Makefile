VENV = .venv312
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.PHONY: help setup setup-gpu run stop pipeline train status clean

help:
	@echo "FaceSim"
	@echo ""
	@echo "  make setup      — install everything for CPU (web demo + pipeline)"
	@echo "  make setup-gpu  — install everything for GPU server (adds PyTorch CUDA + nnUNet)"
	@echo "  make run        — start web demo server at :8000"
	@echo "  make pipeline   — run research pipeline (Phase 1-6)"
	@echo "  make train      — run Phase 6 ML training only (GPU)"
	@echo "  make status     — show pipeline state"
	@echo "  make stop       — stop web server"
	@echo "  make clean      — remove sessions and pipeline outputs"

# ── Setup ────────────────────────────────────────────────────────────

setup:
	@echo "==> Creating Python 3.12 venv..."
	python3.12 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "==> Created .env — edit it: set DEMO_PASSWORD and SEGMENTATION_DEVICE"; \
	else \
		echo "==> .env already exists"; \
	fi
	@echo ""
	@echo "Done. Next: edit .env, then make run (demo) or make pipeline (research)"

setup-gpu: setup
	@echo "==> Installing PyTorch with CUDA 12.1..."
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	@echo "==> Installing nnU-Net v2..."
	$(PIP) install nnunetv2
	@echo "==> Setting nnUNet paths in .env..."
	@grep -q "nnUNet_raw" .env || printf '\n# nnU-Net paths (Phase 6)\nnnUNet_raw=%s/data/nnunet/raw\nnnUNet_preprocessed=%s/data/nnunet/preprocessed\nnnUNet_results=%s/data/nnunet/results\n' "$$PWD" "$$PWD" "$$PWD" >> .env
	@echo "==> GPU check..."
	$(PYTHON) -c "import torch; print('CUDA:', torch.cuda.is_available()); \
		[print('GPU:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
	@echo ""
	@echo "GPU setup done. Edit .env: set SEGMENTATION_DEVICE=cuda"
	@echo "Then: make pipeline  (or make train for Phase 6 only)"

# ── Run ──────────────────────────────────────────────────────────────

run:
	@if [ ! -f .env ]; then echo "ERROR: .env not found. Run: make setup"; exit 1; fi
	@grep -q "change_this_password" .env && echo "ERROR: Change DEMO_PASSWORD in .env first" && exit 1 || true
	@echo "==> Starting FaceSim server at http://0.0.0.0:8000 ..."
	cd server && ../$(VENV)/bin/uvicorn main:app --host 0.0.0.0 --port 8000

stop:
	@pkill -f "uvicorn main:app" && echo "Server stopped" || echo "Server not running"

# ── Research pipeline ────────────────────────────────────────────────

pipeline:
	@if [ ! -f .env ]; then echo "ERROR: .env not found. Run: make setup"; exit 1; fi
	@set -a; . ./.env; set +a; $(PYTHON) pipeline/main.py

train:
	@if [ ! -f .env ]; then echo "ERROR: .env not found. Run: make setup-gpu"; exit 1; fi
	@set -a; . ./.env; set +a; $(PYTHON) pipeline/main.py --phase 6

status:
	@$(PYTHON) pipeline/main.py --status

# ── Clean ─────────────────────────────────────────────────────────────

clean:
	@echo "==> Cleaning sessions and pipeline outputs..."
	rm -rf server/sessions/*
	rm -rf data/mesh data/renders data/sim data/validation
	rm -f pipeline_state.json
	@echo "Done (data/stl and data/seg preserved)"

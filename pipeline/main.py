#!/usr/bin/env python
"""
FaceSim research pipeline — Phase 1→6 orchestrator.

Runs all phases in order with checkpoint/resume, fallbacks, and notifications.
Safe to leave on a server: each phase is idempotent, crashes are logged.

Usage:
  python pipeline/main.py                        # run all phases
  python pipeline/main.py --status               # show state and exit
  python pipeline/main.py --phase 2              # run only phase 2
  python pipeline/main.py --reset-phase 2        # re-run phase 2
  python pipeline/main.py --download toothfairy2 # download dataset
  python pipeline/main.py --list-datasets

Notifications (optional):
  export TELEGRAM_BOT_TOKEN=...
  export TELEGRAM_CHAT_ID=...
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.state import State
from pipeline import notify
from pipeline.phases.p1_seg import Phase1Seg
from pipeline.phases.p2_mesh import Phase2Mesh
from pipeline.phases.p3_sim import Phase3Sim
from pipeline.phases.p4_render import Phase4Render
from pipeline.phases.p5_validate import Phase5Validate
from pipeline.phases.p6_train import Phase6Train

DATA_DIR = ROOT / "data"
STATE_FILE = ROOT / "pipeline_state.json"

ALL_PHASES = [Phase1Seg, Phase2Mesh, Phase3Sim, Phase4Render, Phase5Validate, Phase6Train]


def run(data_dir: Path, state: State, only_phase: int | None = None):
    phases = ALL_PHASES
    if only_phase is not None:
        phases = [p for p in ALL_PHASES if p.name.startswith(f"phase{only_phase}_")]
        if not phases:
            print(f"No phase {only_phase}. Valid: 1-6")
            return

    completed, waiting = [], []

    for PhaseClass in phases:
        phase = PhaseClass()
        success = phase.execute(state, data_dir)
        (completed if success else waiting).append(phase.name)

    print("\n" + "=" * 50)
    print("PIPELINE STATUS")
    print("=" * 50)
    print(state.summary())
    print()

    if completed:
        notify.send(f"✅ FaceSim: completed {completed}")
    if waiting:
        notify.send(f"⏳ FaceSim: waiting on {waiting} — check pipeline_state.json")


def main():
    ap = argparse.ArgumentParser(description="FaceSim research pipeline")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--state", default=str(STATE_FILE))
    ap.add_argument("--status", action="store_true", help="Print state and exit")
    ap.add_argument("--phase", type=int, help="Run only this phase number (1-6)")
    ap.add_argument("--reset-phase", type=int, metavar="N", help="Clear phase N from state to re-run")
    ap.add_argument("--download", choices=["toothfairy2", "toothfairy3", "han_seg"])
    ap.add_argument("--list-datasets", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    state = State(Path(args.state))

    if args.list_datasets:
        from pipeline.data.download import list_datasets
        list_datasets()
        return

    if args.download:
        from pipeline.data.download import download
        download(args.download, data_dir / "raw" / "datasets" / args.download)
        return

    if args.status:
        print(state.summary())
        return

    if args.reset_phase is not None:
        prefix = f"phase{args.reset_phase}_"
        removed = [k for k in list(state.data["phases"]) if k.startswith(prefix)]
        for k in removed:
            del state.data["phases"][k]
        state.save()
        print(f"Reset: {removed or '(nothing to reset)'}")
        return

    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}")
        sys.exit(1)

    run(data_dir, state, only_phase=args.phase)


if __name__ == "__main__":
    main()

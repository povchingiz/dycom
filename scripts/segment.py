"""
Phase 1 segmentation pipeline.
Runs TotalSegmentator tasks: teeth, craniofacial_structures, tissue_types.
Input:  data/nifti/patient.nii.gz
Output: data/seg/{task}/*.nii.gz (per-structure masks)
"""
import argparse
import pathlib
import sys
from totalsegmentator.python_api import totalsegmentator

TASKS = {
    "teeth": "data/seg/teeth",
    "craniofacial_structures": "data/seg/craniofacial",
    "tissue_types": "data/seg/tissue",
}


def run(input_path: str, tasks: list[str], device: str = "cpu") -> None:
    inp = pathlib.Path(input_path)
    if not inp.exists():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    for task in tasks:
        out = pathlib.Path(TASKS[task])
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running task: {task} -> {out} ===")
        totalsegmentator(inp, out, task=task, device=device, quiet=False)
        print(f"=== Done: {task} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="data/nifti/patient.nii.gz")
    parser.add_argument("-t", "--tasks", nargs="+",
                        default=list(TASKS.keys()))
    parser.add_argument("-d", "--device", default="cpu")
    args = parser.parse_args()
    run(args.input, args.tasks, args.device)

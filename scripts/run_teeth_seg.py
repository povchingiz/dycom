from totalsegmentator.python_api import totalsegmentator
import pathlib

if __name__ == "__main__":
    inp = pathlib.Path("data/nifti/patient.nii.gz")
    out = pathlib.Path("data/seg/teeth")
    out.mkdir(parents=True, exist_ok=True)

    print("Starting teeth segmentation (CPU)...")
    totalsegmentator(inp, out, task="teeth", device="cpu", quiet=False, verbose=False)
    print("Teeth segmentation DONE")

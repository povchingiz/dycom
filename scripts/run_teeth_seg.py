"""
Teeth and jawbone segmentation using TotalSegmentator.
"""
from totalsegmentator.python_api import totalsegmentator
import pathlib
from typing import Optional, Callable


def segment_teeth(
    input_path: str,
    output_dir: str,
    device: str = "cuda",
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> None:
    """
    Segment teeth and jawbones from NIfTI volume.
    
    Args:
        input_path: Path to input NIfTI file
        output_dir: Directory for output masks
        device: "cuda" for GPU, "cpu" for CPU
        progress_callback: Optional callback function(message, percentage)
    """
    inp = pathlib.Path(input_path)
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(f"Starting segmentation on {device}...", 10)
    
    print(f"Starting teeth segmentation ({device})...")
    
    # TotalSegmentator has internal progress - we just wrap it
    try:
        totalsegmentator(inp, out, task="teeth", device=device, quiet=False, verbose=False)
        if progress_callback:
            progress_callback("Segmentation complete", 100)
        print("Teeth segmentation DONE")
    except Exception as e:
        if progress_callback:
            progress_callback(f"Segmentation failed: {str(e)}", 0)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="data/nifti/patient.nii.gz")
    parser.add_argument("-o", "--output", default="data/seg/teeth")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    segment_teeth(args.input, args.output, args.device)

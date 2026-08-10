"""
Post-processing for segmentation predictions.

`split_canal_sides` fixes the one thing the 3d_fullres model cannot do: tell the
left inferior alveolar canal from the right one.

Measured on the 37 held-out cases of Dataset114:

    canal union (3|4) Dice           0.8865   ← the network finds both canals
    the network's own side labels    0.607 / 0.281
    geometric side split             0.890 / 0.883

The network detects the canals almost perfectly — predicted total canal volume is
within a few percent of ground truth — and then puts ~80% of both into class 3.
That is not a resolution problem, so more epochs will not fix it: a 3d_fullres
patch around one canal is indistinguishable from the mirrored patch around the
other, and the patch is far too small to see which side of the head it sits on.
The 3d_lowres model got balanced (0.63/0.67) canals precisely because its patches
covered enough of the skull to infer the side.

Left/right is geometry, not texture. Splitting the predicted canal mask by the
mid-sagittal plane recovers the full union score on both sides, for free.
"""
from __future__ import annotations

import numpy as np

# Source labels in the 7-class grouped scheme (see p6_train.GROUPED_LABELS).
MANDIBLE = 1
LEFT_CANAL = 3
RIGHT_CANAL = 4

# Which side of the mid-sagittal plane ToothFairy2 calls "left_canal" (label 3),
# expressed along the derived lateral axis. Verified across all 37 held-out cases.
# NB: this is the dataset's convention, and it disagrees with TotalSegmentator's —
# see ROADMAP "Canal side naming conflict". Flip it for a differently-labelled set.
TF2_LEFT_IS_LOWER = True


def clean_prediction(labels: np.ndarray) -> np.ndarray:
    """Full post-process for a 7-class prediction: drop spurious blobs, then fix
    canal sides. This is what callers should use.

    Why the blob removal matters: ToothFairy2 is dental-FOV CBCT, while a clinical
    scan (Planmeca ProMax here) is a whole head with neck. On the patient the model
    emitted 310 mandible components totalling 103 cc against TotalSegmentator's
    52 cc — the real mandible plus 37 cc of temporal bone and cervical spine, i.e.
    bone it had never seen and confidently mislabelled. Agreement with the incumbent
    on the mandible: 0.604 raw, 0.814 after keeping the largest component.

    The model wins its in-domain benchmark by a wide margin and still needs this:
    a held-out score on the training distribution says nothing about a scan whose
    field of view is twice as large.
    """
    out = keep_largest_components(labels)
    return split_canal_sides(out)


# Anatomy is one mandible, one maxilla, two canals. Teeth are deliberately absent:
# each tooth is its own component and the count varies per patient.
SINGLE_INSTANCE = {MANDIBLE: 1, 2: 1}       # 2 = maxilla
CANAL_COMPONENTS = 2


def keep_largest_components(labels: np.ndarray) -> np.ndarray:
    """Drop anatomically impossible extra components, per class."""
    from scipy.ndimage import label as cc_label

    out = labels.copy()
    for cls, keep_n in SINGLE_INSTANCE.items():
        _prune(out, labels == cls, cls, keep_n, cc_label)

    canal = np.isin(labels, (LEFT_CANAL, RIGHT_CANAL))
    if canal.any():
        # Prune the union, not each side: the network's side labels are unreliable
        # (that is what split_canal_sides exists for), so per-class pruning would
        # throw away a real canal that happened to be labelled with the wrong side.
        components, n = cc_label(canal)
        if n > CANAL_COMPONENTS:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            keep = set(np.argsort(sizes)[-CANAL_COMPONENTS:].tolist())
            drop = canal & ~np.isin(components, list(keep))
            out[drop] = 0
    return out


def _prune(out: np.ndarray, mask: np.ndarray, cls: int, keep_n: int, cc_label) -> None:
    if not mask.any():
        return
    components, n = cc_label(mask)
    if n <= keep_n:
        return
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    keep = np.argsort(sizes)[-keep_n:]
    out[mask & ~np.isin(components, keep)] = 0


def split_canal_sides(labels: np.ndarray,
                      lateral_axis: int | None = None,
                      left_is_lower: bool = TF2_LEFT_IS_LOWER) -> np.ndarray:
    """Reassign canal voxels to left/right by the mid-sagittal plane.

    Args:
        labels: predicted label volume (modified copy is returned)
        lateral_axis: array axis running left-right. Derived from the data when
            None — do not hardcode it, the axis order depends on the export.
        left_is_lower: True if the "left" canal sits at lower coordinates along
            `lateral_axis`.

    Returns the corrected label volume. A volume with no canal voxels, or no
    mandible to place the plane with, is returned unchanged.
    """
    out = labels.copy()
    canal = np.isin(labels, (LEFT_CANAL, RIGHT_CANAL))
    if not canal.any():
        return out

    idx = np.argwhere(canal)
    if lateral_axis is None:
        lateral_axis = _derive_lateral_axis(canal, idx)
        if lateral_axis is None:
            return out

    mid = _midsagittal(labels, idx, lateral_axis)
    lower = idx[idx[:, lateral_axis] < mid]
    upper = idx[idx[:, lateral_axis] >= mid]

    left_side, right_side = (lower, upper) if left_is_lower else (upper, lower)
    out[canal] = 0
    if len(left_side):
        out[tuple(left_side.T)] = LEFT_CANAL
    if len(right_side):
        out[tuple(right_side.T)] = RIGHT_CANAL
    return out


def _derive_lateral_axis(canal: np.ndarray, idx: np.ndarray) -> int | None:
    """Axis separating the two canals.

    Preferred cue: the two connected components of the canal mask sit on opposite
    sides of the head, so the axis of their centroid difference IS the lateral
    axis. Falls back to the canal cloud's widest extent when the components merge
    or only one is found."""
    try:
        from scipy.ndimage import label as cc_label
        components, n = cc_label(canal)
        if n >= 2:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            first, second = np.argsort(sizes)[-2:]
            c1 = np.argwhere(components == first).mean(axis=0)
            c2 = np.argwhere(components == second).mean(axis=0)
            return int(np.argmax(np.abs(c1 - c2)))
    except ImportError:
        pass
    extent = idx.max(axis=0) - idx.min(axis=0)
    return int(np.argmax(extent)) if extent.max() > 0 else None


def _midsagittal(labels: np.ndarray, canal_idx: np.ndarray, axis: int) -> float:
    """Mid-sagittal position along `axis`.

    Taken from the mandible: it is a symmetric structure covering the whole width
    of the face, so its centroid lands on the midline. The canal cloud's own
    midpoint is the fallback, which is worse — an asymmetric prediction drags it
    off centre, which is exactly the case being corrected."""
    mandible = np.argwhere(labels == MANDIBLE)
    if len(mandible):
        return float(mandible.mean(axis=0)[axis])
    return float((canal_idx[:, axis].min() + canal_idx[:, axis].max()) / 2.0)

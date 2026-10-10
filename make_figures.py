"""Make visual evidence: a before/after grid and failure-case crops.

Usage:
    python make_figures.py --input data/ --output figures/

Outputs:
    figures/grid.png            original | classical (| SR if the model exists)
    figures/failure_<n>_*.png   reference | degraded | classical crops for the
                                3 tiles where the classical pipeline lost the most PSNR
"""
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from enhance import IMAGE_EXTS, classical_enhance, foreground_mask, load_sr_model, sr_upscale
from evaluate import degrade_fullres, masked_metrics, SEED


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Put a small caption bar on top of an image."""
    bar = np.full((22, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit(img: np.ndarray, size: int = 300) -> np.ndarray:
    """Resize to fit inside a size x size square, padded with white."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    canvas[: small.shape[0], : small.shape[1]] = small
    return canvas


def make_grid(files: List[Path], out_path: Path, sr, n: int = 6) -> None:
    rows = []
    for f in files[:n]:
        img = cv2.imread(str(f))
        cols = [label(fit(img), "original"), label(fit(classical_enhance(img)), "classical")]
        if sr is not None:
            cols.append(label(fit(sr_upscale(img, sr)), "SR x2"))
        rows.append(np.hstack(cols))
    cv2.imwrite(str(out_path), np.vstack(rows))


def worst_crop(ref: np.ndarray, a: np.ndarray, b: np.ndarray, mask: np.ndarray, size: int = 80) -> Tuple[int, int]:
    """Top-left corner of the window where b differs most from ref (inside the roof)."""
    diff = np.abs(ref.astype(np.float32) - b.astype(np.float32)).mean(axis=2) * mask
    k = cv2.boxFilter(diff, -1, (size, size))
    h, w = diff.shape
    k[: size // 2] = 0; k[h - size // 2:] = 0; k[:, : size // 2] = 0; k[:, w - size // 2:] = 0
    y, x = np.unravel_index(np.argmax(k), k.shape)
    return max(0, y - size // 2), max(0, x - size // 2)


def make_failures(files: List[Path], out_dir: Path, k: int = 3) -> None:
    scored = []
    for i, f in enumerate(files):
        ref = cv2.imread(str(f))
        h, w = ref.shape[:2]
        ref = ref[: h // 2 * 2, : w // 2 * 2]
        mask = foreground_mask(ref)
        deg = degrade_fullres(ref, np.random.default_rng(SEED + i))
        out = classical_enhance(deg, mask=mask)
        drop = masked_metrics(ref, deg, mask)["psnr"] - masked_metrics(ref, out, mask)["psnr"]
        scored.append((drop, f, ref, deg, out, mask))
    scored.sort(key=lambda t: -t[0])
    for n, (drop, f, ref, deg, out, mask) in enumerate(scored[:k], 1):
        y, x = worst_crop(ref, deg, out, mask)
        crops = [label(cv2.resize(im[y:y + 80, x:x + 80], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), t)
                 for im, t in ((ref, "reference"), (deg, "degraded"), (out, "classical"))]
        path = out_dir / f"failure_{n}_{f.stem.replace('building_', '')}.png"
        cv2.imwrite(str(path), np.hstack(crops))
        print(f"{path.name}: classical lost {drop:.2f} dB PSNR vs degraded input")


def main() -> None:
    p = argparse.ArgumentParser(description="Create before/after grid and failure crops.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="figures")
    p.add_argument("--source", default="google")
    args = p.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in Path(args.input).iterdir()
                   if f.suffix.lower() in IMAGE_EXTS and args.source in f.name)
    if not files:
        raise SystemExit("No matching images found.")
    sr: Optional[object] = None
    try:
        sr = load_sr_model()
    except (FileNotFoundError, AttributeError, cv2.error):
        print("SR model not found - grid will show original vs classical only.")
    make_grid(files, out_dir / "grid.png", sr)
    make_failures(files, out_dir)
    print(f"Figures saved in {out_dir}/")


if __name__ == "__main__":
    main()

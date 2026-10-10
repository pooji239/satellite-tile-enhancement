"""Evaluation: do SR and classical enhancement actually help?

Usage:
    python evaluate.py --input data/ --report results.md

Idea: there is no ground truth, so we MANUFACTURE one. Each tile is treated as
the reference. We degrade it on purpose, run each method on the degraded copy,
and compare the result back to the reference (PSNR / SSIM, foreground only).

  Experiment A (upscaling): reference -> downscale 2x + JPEG + noise = LR.
      Methods: bicubic 2x vs pretrained FSRCNN 2x. Output is same size as ref.
  Experiment B (classical): reference -> blur + JPEG + noise = degraded.
      Compare degraded vs classical(degraded) against the reference.
      If classical is better than "do nothing", it helped; otherwise it hurt.
  Real tiles (no reference): Laplacian variance (sharpness) and colour shift
      between each real tile and its classical output.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from enhance import (IMAGE_EXTS, bicubic_upscale, classical_enhance,
                     foreground_mask, load_sr_model, sr_upscale)

SEED = 0
JPEG_QUALITY = 70
NOISE_SIGMA = 2.0


def jpeg_roundtrip(img: np.ndarray, quality: int = JPEG_QUALITY) -> np.ndarray:
    """Compress to JPEG and decode again (adds real JPEG artefacts)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def add_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add mild Gaussian sensor-like noise."""
    noisy = img.astype(np.float32) + rng.normal(0, NOISE_SIGMA, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def degrade_lowres(ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Experiment A input: 2x downscale (area) + noise + JPEG."""
    h, w = ref.shape[:2]
    lr = cv2.resize(ref, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return jpeg_roundtrip(add_noise(lr, rng))


def degrade_fullres(ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Experiment B input: slight blur + noise + JPEG, same size."""
    blurred = cv2.GaussianBlur(ref, (0, 0), sigmaX=1.0)
    return jpeg_roundtrip(add_noise(blurred, rng))


def masked_metrics(ref: np.ndarray, test: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """PSNR and SSIM using foreground (roof) pixels only."""
    mask = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if mask.sum() == 0:
        return {"psnr": float("nan"), "ssim": float("nan")}
    mse = np.mean((ref[mask].astype(np.float64) - test[mask].astype(np.float64)) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    _, ssim_map = structural_similarity(ref, test, channel_axis=2, full=True, data_range=255)
    return {"psnr": psnr, "ssim": float(ssim_map[mask].mean())}


def sharpness(img: np.ndarray, mask: np.ndarray) -> float:
    """No-reference sharpness: variance of the Laplacian on foreground pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return float(lap[mask].var()) if mask.any() else float("nan")


def colour_shift(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean abs change in LAB a/b channels (colour only) - spots colour casts."""
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 1:]
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 1:]
    return float(np.abs(la[mask] - lb[mask]).mean())


def fmt(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def evaluate(files: List[Path], sr) -> List[Dict[str, float]]:
    rows = []
    for i, path in enumerate(files):
        ref = cv2.imread(str(path))
        if ref is None:
            continue
        h, w = ref.shape[:2]
        ref = ref[: h // 2 * 2, : w // 2 * 2]  # even size so 2x round-trips exactly
        mask = foreground_mask(ref)
        rng = np.random.default_rng(SEED + i)
        row: Dict[str, float] = {"name": path.stem.replace("building_", ""),
                                 "size": f"{ref.shape[1]}x{ref.shape[0]}"}

        # Experiment A: upscaling
        lr = degrade_lowres(ref, rng)
        outputs = {"bicubic": bicubic_upscale(lr)}
        if sr is not None:
            outputs["sr"] = sr_upscale(lr, sr)
        for name, out in outputs.items():
            m = masked_metrics(ref, out, mask)
            row[f"A_{name}_psnr"], row[f"A_{name}_ssim"] = m["psnr"], m["ssim"]

        # Experiment B: classical vs doing nothing (+ ablation without tone step)
        deg = degrade_fullres(ref, rng)
        fixed = classical_enhance(deg, mask=mask)
        no_tone = classical_enhance(deg, mask=mask, clahe_clip=0)
        for name, out in {"none": deg, "classical": fixed, "notone": no_tone}.items():
            m = masked_metrics(ref, out, mask)
            row[f"B_{name}_psnr"], row[f"B_{name}_ssim"] = m["psnr"], m["ssim"]

        # Real tile, no reference
        real_out = classical_enhance(ref, mask=mask)
        row["sharp_orig"] = sharpness(ref, mask)
        row["sharp_cls"] = sharpness(real_out, mask)
        row["colour_shift"] = colour_shift(ref, real_out, mask)
        rows.append(row)
        print(f"[{i + 1}/{len(files)}] {path.name}")
    return rows


def mean(rows: List[Dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def write_report(rows: List[Dict], report: Path, has_sr: bool) -> None:
    L: List[str] = ["# Evaluation results", "",
                    f"Tiles: {len(rows)}. Degradation: JPEG q={JPEG_QUALITY}, "
                    f"noise sigma={NOISE_SIGMA}, seed={SEED}. Metrics use roof pixels only "
                    "(grey background masked out). Higher PSNR/SSIM = closer to the original.", ""]

    L += ["## Experiment A - 2x upscaling (bicubic vs pretrained FSRCNN)", ""]
    if has_sr:
        L += ["| Tile | Size | Bicubic PSNR | SR PSNR | Bicubic SSIM | SR SSIM |", "|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['name']} | {r['size']} | {fmt(r['A_bicubic_psnr'])} | {fmt(r['A_sr_psnr'])} "
                     f"| {fmt(r['A_bicubic_ssim'], 3)} | {fmt(r['A_sr_ssim'], 3)} |")
        L.append(f"| **Mean** | | **{fmt(mean(rows, 'A_bicubic_psnr'))}** | **{fmt(mean(rows, 'A_sr_psnr'))}** "
                 f"| **{fmt(mean(rows, 'A_bicubic_ssim'), 3)}** | **{fmt(mean(rows, 'A_sr_ssim'), 3)}** |")
        wins = sum(r["A_sr_psnr"] > r["A_bicubic_psnr"] for r in rows)
        L += ["", f"SR beats bicubic on PSNR in {wins}/{len(rows)} tiles."]
    else:
        L += ["SR model not found (models/FSRCNN_x2.pb) - only bicubic was run.", "",
              "| Tile | Size | Bicubic PSNR | Bicubic SSIM |", "|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['name']} | {r['size']} | {fmt(r['A_bicubic_psnr'])} | {fmt(r['A_bicubic_ssim'], 3)} |")
        L.append(f"| **Mean** | | **{fmt(mean(rows, 'A_bicubic_psnr'))}** | **{fmt(mean(rows, 'A_bicubic_ssim'), 3)}** |")

    L += ["", "## Experiment B - classical pipeline vs doing nothing", "",
          "`notone` = same pipeline with the CLAHE contrast step switched off (ablation).", "",
          "| Tile | Degraded PSNR | Classical PSNR | No-tone PSNR | Degraded SSIM | Classical SSIM | No-tone SSIM |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {fmt(r['B_none_psnr'])} | {fmt(r['B_classical_psnr'])} | {fmt(r['B_notone_psnr'])} "
                 f"| {fmt(r['B_none_ssim'], 3)} | {fmt(r['B_classical_ssim'], 3)} | {fmt(r['B_notone_ssim'], 3)} |")
    L.append(f"| **Mean** | **{fmt(mean(rows, 'B_none_psnr'))}** | **{fmt(mean(rows, 'B_classical_psnr'))}** "
             f"| **{fmt(mean(rows, 'B_notone_psnr'))}** | **{fmt(mean(rows, 'B_none_ssim'), 3)}** "
             f"| **{fmt(mean(rows, 'B_classical_ssim'), 3)}** | **{fmt(mean(rows, 'B_notone_ssim'), 3)}** |")
    for key, label in (("classical", "Full classical"), ("notone", "No-tone variant")):
        helped = sum(r[f"B_{key}_psnr"] > r["B_none_psnr"] for r in rows)
        L.append(f"\n{label} improved PSNR in {helped}/{len(rows)} tiles (hurt in {len(rows) - helped}).")
    L += ["", "Caveat: the degradation includes a blur, which favours sharpening; the reference tiles are "
          "already well exposed, so any tone change counts as 'error' to PSNR even if it helps a hazy tile."]

    L += ["", "## Real tiles, no reference (classical output vs original)", "",
          "| Tile | Sharpness before | Sharpness after | Colour shift (LAB a/b) |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {fmt(r['sharp_orig'], 1)} | {fmt(r['sharp_cls'], 1)} | {fmt(r['colour_shift'])} |")
    L.append(f"| **Mean** | **{fmt(mean(rows, 'sharp_orig'), 1)}** | **{fmt(mean(rows, 'sharp_cls'), 1)}** "
             f"| **{fmt(mean(rows, 'colour_shift'))}** |")
    L += ["", "Higher sharpness is not automatically better: noise and halos also raise it."]
    report.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate enhancement methods.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", default="results.md")
    parser.add_argument("--source", default="google",
                        help="Only use files whose name contains this (default: google). Use '' for all.")
    args = parser.parse_args()

    files = sorted(p for p in Path(args.input).iterdir()
                   if p.suffix.lower() in IMAGE_EXTS and args.source in p.name)
    if not files:
        raise SystemExit("No matching images found.")
    sr: Optional[object] = None
    try:
        sr = load_sr_model()
    except (FileNotFoundError, AttributeError, cv2.error) as e:
        print(f"Warning: SR disabled ({e})")
    rows = evaluate(files, sr)
    write_report(rows, Path(args.report), has_sr=sr is not None)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()

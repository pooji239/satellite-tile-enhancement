"""Satellite tile enhancement CLI.

Usage:
    python enhance.py --input data/ --output out/ --method classical
    python enhance.py --input data/ --output out/ --method sr
    python enhance.py --input data/ --output out/ --method bicubic   (baseline)

Dataset note: the images are building crops with a flat grey background
(BGR 30,30,30). That background is NOT image content, so it is masked out.
"""
import argparse
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
BG_COLOUR = 30  # flat grey used as background in the dataset
MODEL_PATH = Path(__file__).parent / "models" / "FSRCNN_x2.pb"


def foreground_mask(img: np.ndarray) -> np.ndarray:
    """Boolean mask, True where the pixel is real image content (not grey bg)."""
    return ~np.all(img == BG_COLOUR, axis=2)


def classical_enhance(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    denoise_strength: int = 2,
    clahe_clip: float = 1.0,
    sharpen_amount: float = 0.8,
) -> np.ndarray:
    """Classical (no learned model) enhancement of a BGR uint8 image.

    Order matters:
      1. Denoise FIRST - otherwise contrast boost and sharpening amplify
         JPEG blocking and noise.
      2. Contrast (CLAHE) on the L channel only (LAB) - keeps colours stable.
      3. Sharpen LAST (unsharp mask) - sharpen real edges, not noise.
    Set any strength to 0 to switch that step off (used for ablations).
    The grey background is restored afterwards so it never changes.
    """
    if mask is None:
        mask = foreground_mask(img)

    out = img.copy()
    if denoise_strength > 0:
        out = cv2.fastNlMeansDenoisingColored(
            out, None, h=denoise_strength, hColor=denoise_strength,
            templateWindowSize=7, searchWindowSize=21,
        )

    if clahe_clip > 0:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        out = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    if sharpen_amount > 0:
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.5)
        out = cv2.addWeighted(out, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
    out = np.clip(out, 0, 255).astype(np.uint8)

    out[~mask] = img[~mask]  # put the untouched background back
    return out


def bicubic_upscale(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Baseline: plain bicubic resize."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def load_sr_model(model_path: Path = MODEL_PATH):
    """Load the pretrained FSRCNN x2 model (needs opencv-contrib-python)."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"{model_path} not found. Download FSRCNN_x2.pb into the models/ folder "
            "(see README)."
        )
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel("fsrcnn", 2)
    return sr


def sr_upscale(img: np.ndarray, sr=None) -> np.ndarray:
    """2x learned super-resolution (pretrained FSRCNN, nothing trained here)."""
    if sr is None:
        sr = load_sr_model()
    return sr.upsample(img)


def get_method(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function img -> enhanced img for the given method name."""
    if name == "classical":
        return classical_enhance
    if name == "bicubic":
        return bicubic_upscale
    if name == "sr":
        sr = load_sr_model()
        return lambda img: sr_upscale(img, sr)
    raise ValueError(f"Unknown method: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance satellite tiles.")
    parser.add_argument("--input", required=True, help="Folder of input images")
    parser.add_argument("--output", required=True, help="Folder for results")
    parser.add_argument("--method", choices=["classical", "sr", "bicubic"],
                        default="classical")
    parser.add_argument("--denoise", type=int, default=2, help="NLM strength (0=off)")
    parser.add_argument("--clahe", type=float, default=1.0, help="CLAHE clip (0=off)")
    parser.add_argument("--sharpen", type=float, default=0.8, help="Unsharp amount (0=off)")
    args = parser.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.is_dir():
        raise SystemExit(f"Input folder not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = get_method(args.method)
    if args.method == "classical":
        fn = lambda img: classical_enhance(  # noqa: E731
            img, denoise_strength=args.denoise, clahe_clip=args.clahe,
            sharpen_amount=args.sharpen)

    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"No images found in {in_dir}")

    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Skipping unreadable file: {path.name}")
            continue
        result = fn(img)
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), result)  # PNG: no new JPEG loss
        print(f"{path.name}: {img.shape[:2]} -> {result.shape[:2]}")


if __name__ == "__main__":
    main()

# Satellite Tile Enhancement

Classical enhancement, 2x super-resolution (pretrained FSRCNN) and an evaluation
that builds its own reference images. CPU only.

## Setup
```
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python download_model.py       # one-time: fetches models/FSRCNN_x2.pb (~40 KB)
```
Put the dataset images in `data/`.

## Run
```
python enhance.py --input data/ --output out/ --method classical
python enhance.py --input data/ --output out_sr/ --method sr
python enhance.py --input data/ --output out_bicubic/ --method bicubic
python evaluate.py --input data/ --report results.md
python make_figures.py --input data/ --output figures/
```
Optional tuning: `--denoise 2 --clahe 1.0 --sharpen 0.8` (0 switches a step off).

## Assumptions
- The supplied dataset has 41 masked building crops (24 Google, 17 Mapbox), varied sizes, PNG,
  not the 20 x 640x640 JPEG tiles in the brief. Only `*_google.png` files are used
  (`--source ""` includes everything).
- The flat grey background (BGR 30,30,30) is not image content. It is left untouched by
  enhancement and excluded from all metrics.
- No Google API key, so a true `scale=2` reference was not fetched. References are the tiles
  themselves, degraded synthetically (2x downscale / blur, noise sigma 2, JPEG q70).
- The tiles are already compressed, so the "reference" is not perfect. See WRITEUP.md.

## Files
`enhance.py` enhancement CLI | `evaluate.py` evaluation CLI | `make_figures.py` grid and
failure crops | `download_model.py` model fetch | `WRITEUP.md` one-page write-up

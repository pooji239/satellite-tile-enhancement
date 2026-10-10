"""Download the pretrained FSRCNN x2 model into models/ (run once)."""
import urllib.request
from pathlib import Path

URL = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
DEST = Path(__file__).parent / "models" / "FSRCNN_x2.pb"

if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to {DEST} ({DEST.stat().st_size / 1024:.0f} KB)")

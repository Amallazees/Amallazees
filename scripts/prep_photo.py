"""
Prepare a portrait photo for clean ASCII conversion:
  1. remove the background (rembg) so the subject is isolated
  2. boost LOCAL contrast (CLAHE) so a flatly-lit face gains highlights and
     shadows -- this is what turns a dark blob into a recognizable face
  3. composite the subject onto pure white so the background reads as blank
     (white -> spaces in the ascii ramp)

Output: source-prepped.png (grayscale), consumed by make_ascii_svg.py.
Run once whenever the source photo changes; the ascii SVG itself is static.

    python scripts/prep_photo.py <input.jpg> [output.png]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image
from rembg import remove

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.jpg")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

img = cv2.imread(INP)
if img is None:
    raise FileNotFoundError(f"Cannot read image at {INP}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# If the image is a circular silhouette avatar photo
cy, cx = h // 2, w // 2
y, x = np.ogrid[:h, :w]
dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)

silhouette_mask = (gray < 85) & (dist_from_center < 450)
if np.sum(silhouette_mask) > 1000:
    out = np.full_like(gray, 255)
    out[silhouette_mask] = gray[silhouette_mask]
    kernel = np.ones((3, 3), np.uint8)
    out_inv = cv2.bitwise_not(out)
    out_inv = cv2.morphologyEx(out_inv, cv2.MORPH_OPEN, kernel)
    out = cv2.bitwise_not(out_inv)
else:
    cut = remove(Image.open(INP).convert("RGBA"))
    rgb = np.array(cut.convert("RGB"))
    alpha = np.array(cut.split()[-1])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.05, beta=18)
    mask = (alpha.astype(np.float32) / 255.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
    out = gray.astype(np.float32) * mask + 255.0 * (1.0 - mask)
    out = np.clip(out, 0, 255).astype(np.uint8)

Image.fromarray(out, mode="L").save(OUT)
print("wrote", OUT, out.shape)

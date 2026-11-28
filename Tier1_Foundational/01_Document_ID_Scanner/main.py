"""
Document & ID Scanner, command-line version of the notebook.
Same pipeline, same steps, written top to bottom.

Usage:
    python main.py --image samples/your_photo.jpg
    python main.py --image samples/your_photo.jpg --show   # also plot every step
"""
import argparse
import os

import cv2
import matplotlib
import numpy as np

parser = argparse.ArgumentParser(description="Turn an angled phone photo into a flat scan.")
parser.add_argument("--image", required=True, help="path to ONE photo (document or ID)")
parser.add_argument("--output", default="outputs", help="folder for the final scan")
parser.add_argument("--show", action="store_true", help="plot the result of every step")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")  # no window needed when we only want the saved scan
import matplotlib.pyplot as plt

# settings you may want to adjust
DETECT_HEIGHT = 800
CANNY_LOW, CANNY_HIGH = 50, 150
MIN_AREA_RATIO = 0.2
APPROX_EPSILON = 0.02
BLOCK_SIZE, C = 21, 10

# ---- Step 1: load the photo --------------------------------------------------
img = cv2.imread(args.image)
if img is None:
    raise SystemExit(f"Couldn't read '{args.image}'. Is it a valid image file?")
print("Loaded", args.image, "-", img.shape[1], "x", img.shape[0], "px")

if args.show:
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Step 1: original photo")
    plt.axis("off")
    plt.show()

# ---- Step 2: Canny edges (on a shrunken copy, it is faster and cleaner) --------
h, w = img.shape[:2]
ratio = h / DETECT_HEIGHT
small = cv2.resize(img, (int(w / ratio), DETECT_HEIGHT), interpolation=cv2.INTER_AREA)
blurred = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # join broken edge pieces
edges = cv2.dilate(edges, kernel, iterations=1)

if args.show:
    plt.figure(figsize=(8, 8))
    plt.imshow(edges, cmap="gray")
    plt.title("Step 2: Canny edges")
    plt.axis("off")
    plt.show()

# ---- Step 3: contour & corner detection -------------------------------------
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

page_contour = None
for c in contours:
    if cv2.contourArea(c) < MIN_AREA_RATIO * small.shape[0] * small.shape[1]:
        break
    approx = cv2.approxPolyDP(c, APPROX_EPSILON * cv2.arcLength(c, True), True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        page_contour = approx
        break
if page_contour is None:
    raise SystemExit("No 4-corner page found. Adjust the Canny thresholds or MIN_AREA_RATIO, "
                     "or use a plainer background.")

# order as top-left, top-right, bottom-right, bottom-left, then scale to full size
pts = page_contour.reshape(4, 2).astype("float32")
s = pts.sum(axis=1)
d = np.diff(pts, axis=1).ravel()
corners = np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                    pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32") * ratio

if args.show:
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    outline = np.vstack([corners, corners[:1]])
    plt.plot(outline[:, 0], outline[:, 1], "g-", linewidth=3)
    plt.scatter(corners[:, 0], corners[:, 1], c="red", s=80, zorder=3)
    for i, (x, y) in enumerate(corners):
        plt.text(x + 20, y - 20, str(i + 1), color="yellow", fontsize=16, weight="bold")
    plt.title("Step 3: detected page corners (1=TL, 2=TR, 3=BR, 4=BL)")
    plt.axis("off")
    plt.show()

# ---- Step 4: homography + warpPerspective -------------------------------------
tl, tr, br, bl = corners
out_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
out_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
destination = np.array([[0, 0], [out_w - 1, 0],
                        [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
H = cv2.getPerspectiveTransform(corners, destination)
warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_CUBIC)

if args.show:
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Step 4: flattened page")
    plt.axis("off")
    plt.show()

# ---- Step 5: adaptive threshold ------------------------------------------------
gray_warped = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (3, 3), 0)
scanned = cv2.adaptiveThreshold(gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, BLOCK_SIZE, C)

if args.show:
    plt.figure(figsize=(8, 8))
    plt.imshow(scanned, cmap="gray")
    plt.title("Step 5: adaptive threshold")
    plt.axis("off")
    plt.show()

# ---- Step 6: save the final scan -------------------------------------------
os.makedirs(args.output, exist_ok=True)
stem = os.path.splitext(os.path.basename(args.image))[0]
output_path = os.path.join(args.output, f"{stem}_scan.png")
cv2.imwrite(output_path, scanned)
print("Saved to", output_path)

if args.show:
    plt.figure(figsize=(10, 10))
    plt.imshow(scanned, cmap="gray")
    plt.title("Final scan")
    plt.axis("off")
    plt.show()

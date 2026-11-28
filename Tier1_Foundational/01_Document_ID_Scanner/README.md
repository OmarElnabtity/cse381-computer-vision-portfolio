# Automatic Document & ID Scanner

Turns a phone photo of a page or ID card, taken at an angle, into a flat, clean "scanned" image, the CamScanner problem, solved with classical computer vision (no deep learning). It finds the four corners of the page, flattens the perspective with a homography, and applies adaptive thresholding so shadows and uneven lighting don't ruin the result. It is useful for digitizing notes, documents and Arabic textbook pages.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/<repo-name>/blob/main/Tier1_Foundational/01_Document_ID_Scanner/Document_ID_Scanner.ipynb)

![demo](docs/demo.gif)

## Result

Input: `3024 x 4032` px photo → Output: `1139 x 1462` px flat scan

## Pipeline

```
photo → Canny edges → contour / corner detection → homography → warpPerspective → adaptive threshold → scan
```

| Step | What happens |
|---|---|
| 1. Canny edges | shrink, grayscale, blur, Canny, close + dilate to get one closed outline |
| 2. Contour / corner detection | biggest convex 4-point contour = the page; corners ordered TL, TR, BR, BL |
| 3. Homography + warpPerspective | 4 corners are mapped onto a flat rectangle |
| 4. Adaptive threshold | each pixel is compared with its local neighbourhood for the clean scanned look |

## How to run

```bash
git clone https://github.com/OmarElnabtity/computer-vision-portfolio
cd computer-vision-portfolio/Tier1_Foundational/01_Document_ID_Scanner
pip install -r requirements.txt
python main.py --image samples/your_photo.jpg --show
```

The scan is saved to `outputs/`. Leave out `--show` to skip the step-by-step plots. You can also open `Document_ID_Scanner.ipynb` in Google Colab (badge above), upload one photo and run the cells in order.

## Tuning

Every step has a few settings at the top of its notebook cell (also at the top of `main.py`):

| If you see | Change |
|---|---|
| Page outline broken or missing | lower `CANNY_LOW` / `CANNY_HIGH` |
| Wrong object detected | raise `MIN_AREA_RATIO` |
| Text broken or thin | lower `C` or raise `BLOCK_SIZE` |
| Speckles in the background | raise `C` |

## Limitations

- The whole page has to be in the frame, with all four corners visible.
- White paper on a white desk has almost no edge contrast, so use a darker, plain surface.
- A homography assumes a flat page, so curved book pages won't be fully flattened.

## Project structure

```
01_Document_ID_Scanner/
├── Document_ID_Scanner.ipynb
├── main.py
├── requirements.txt
├── samples/
├── outputs/
└── docs/
```

## Privacy note

Don't commit API keys, passwords or personal data. IDs and documents are personal data, so `samples/` and `outputs/` are excluded by `.gitignore` and your photos and scans stay on your machine. Use only non-sensitive images for `docs/demo.gif`.

## What I learned

Blurring and scaling the image correctly is as important as adjusting contrast for edge detection, as background noise has a great influence on how the shape is detected. It is also best that the scanned document takes almost the full picture dimensions to avoid program confusion.

## Built with

Python · OpenCV · NumPy · Matplotlib — Computer Vision, MSA University

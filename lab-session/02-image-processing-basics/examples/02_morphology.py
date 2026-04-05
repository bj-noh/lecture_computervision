from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    image_path = DATA_DIR / "dog1.jpg"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    top = np.hstack([binary, erosion, dilation])
    bottom = np.hstack([opening, closing, np.zeros_like(binary)])
    canvas = np.vstack([top, bottom])

    output_path = OUTPUT_DIR / "morphology_grid.png"
    cv2.imwrite(str(output_path), canvas)

    cv2.imshow("Morphology Grid", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved morphology grid to: {output_path}")


if __name__ == "__main__":
    main()

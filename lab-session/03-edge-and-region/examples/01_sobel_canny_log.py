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
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        sys.exit(f"Could not open image: {image_path}")

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_x = cv2.convertScaleAbs(grad_x)
    sobel_y = cv2.convertScaleAbs(grad_y)
    sobel_mag = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    canny = cv2.Canny(gray, 100, 200)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0.1)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log = np.uint8(np.absolute(log))

    canvas = np.hstack([gray, sobel_mag, canny, log])
    output_path = OUTPUT_DIR / "edge_comparison.png"
    cv2.imwrite(str(output_path), canvas)

    cv2.imshow("Gray | Sobel | Canny | LoG", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved edge comparison to: {output_path}")


if __name__ == "__main__":
    main()

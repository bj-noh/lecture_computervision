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

    _, binary_fixed = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    otsu_threshold, binary_otsu = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    canvas = np.hstack([image, binary_fixed, binary_otsu])
    output_path = OUTPUT_DIR / "threshold_comparison.png"
    cv2.imwrite(str(output_path), canvas)

    cv2.imshow("Gray | Fixed 127 | Otsu", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Otsu threshold: {otsu_threshold:.2f}")
    print(f"Saved comparison to: {output_path}")


if __name__ == "__main__":
    main()

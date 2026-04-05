from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    corrected = np.power(image_float, 1.0 / gamma)
    return np.uint8(np.clip(corrected * 255.0, 0, 255))


def main() -> None:
    image_path = DATA_DIR / "dog1.jpg"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    gamma_dark = gamma_correction(image, gamma=0.5)
    gamma_bright = gamma_correction(image, gamma=2.0)
    equalized = cv2.equalizeHist(image)

    canvas = np.hstack([image, gamma_dark, gamma_bright, equalized])
    output_path = OUTPUT_DIR / "gamma_histogram_comparison.png"
    cv2.imwrite(str(output_path), canvas)

    cv2.imshow("Original | Gamma 0.5 | Gamma 2.0 | Equalized", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved tone comparison to: {output_path}")


if __name__ == "__main__":
    main()

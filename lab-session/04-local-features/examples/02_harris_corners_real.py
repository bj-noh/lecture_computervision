from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    image_path = DATA_DIR / "view.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray32 = np.float32(gray)
    response = cv2.cornerHarris(gray32, blockSize=2, ksize=3, k=0.04)
    response = cv2.dilate(response, None)

    overlay = image.copy()
    threshold = 0.01 * response.max()
    overlay[response > threshold] = [0, 0, 255]

    response_norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    response_color = cv2.applyColorMap(response_norm, cv2.COLORMAP_JET)
    merged = np.hstack([image, response_color, overlay])

    output_path = OUTPUT_DIR / "harris_real_image.png"
    cv2.imwrite(str(output_path), merged)

    cv2.imshow("Original | Harris Response | Corners", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Corner threshold: {threshold:.4f}")
    print(f"Saved Harris visualization to: {output_path}")


if __name__ == "__main__":
    main()

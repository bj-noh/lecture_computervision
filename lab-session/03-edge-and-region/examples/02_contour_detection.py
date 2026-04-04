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
    image = cv2.imread(str(image_path))

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contour_image = image.copy()
    long_contours = [cnt for cnt in contours if cnt.shape[0] > 100]
    cv2.drawContours(contour_image, long_contours, -1, (0, 255, 0), 2)

    merged = np.hstack([image, cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR), contour_image])
    output_path = OUTPUT_DIR / "contour_detection.png"
    cv2.imwrite(str(output_path), merged)

    cv2.imshow("Original | Canny | Contours", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Detected contours: {len(long_contours)}")
    print(f"Saved contour visualization to: {output_path}")


if __name__ == "__main__":
    main()

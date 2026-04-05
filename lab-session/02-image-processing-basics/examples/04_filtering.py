from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    image_path = DATA_DIR / "view.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    average_kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    average_blur = cv2.filter2D(image, -1, average_kernel)
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

    emboss_kernel = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    embossed = cv2.filter2D(image.astype(np.int16), -1, emboss_kernel)
    embossed = np.uint8(np.clip(embossed + 128, 0, 255))

    canvas = np.hstack([image, average_blur, gaussian_blur, embossed])
    output_path = OUTPUT_DIR / "filtering_comparison.png"
    cv2.imwrite(str(output_path), canvas)

    cv2.imshow("Original | Average | Gaussian | Emboss", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved filtering comparison to: {output_path}")


if __name__ == "__main__":
    main()

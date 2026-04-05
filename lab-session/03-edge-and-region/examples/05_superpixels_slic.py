from pathlib import Path

import cv2
import numpy as np
from skimage import data, segmentation


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    image = data.coffee()
    slic_a = segmentation.slic(image, compactness=20, n_segments=400, start_label=1)
    slic_b = segmentation.slic(image, compactness=40, n_segments=400, start_label=1)

    marked_a = np.uint8(segmentation.mark_boundaries(image, slic_a) * 255.0)
    marked_b = np.uint8(segmentation.mark_boundaries(image, slic_b) * 255.0)

    merged = np.hstack([image, marked_a, marked_b])
    merged_bgr = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
    output_path = OUTPUT_DIR / "slic_superpixels.png"
    cv2.imwrite(str(output_path), merged_bgr)

    cv2.imshow("Original | SLIC c=20 | SLIC c=40", merged_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved SLIC result to: {output_path}")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def tile_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def main() -> None:
    image_path = DATA_DIR / "cat.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    h, w = image.shape[:2]

    translation = np.float32([[1, 0, 60], [0, 1, 40]])
    translated = cv2.warpAffine(image, translation, (w, h))

    scaled = cv2.resize(image, dsize=(0, 0), fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
    scaled = scaled[:h, :w]
    if scaled.shape[0] < h or scaled.shape[1] < w:
        padded = np.zeros_like(image)
        padded[: scaled.shape[0], : scaled.shape[1]] = scaled
        scaled = padded

    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1.0)
    rotated = cv2.warpAffine(image, rotation, (w, h))

    reflected = cv2.flip(image, 1)

    panel_size = (320, 240)
    tiled = np.vstack(
        [
            np.hstack([tile_resize(image, panel_size), tile_resize(translated, panel_size)]),
            np.hstack([tile_resize(rotated, panel_size), tile_resize(reflected, panel_size)]),
        ]
    )

    output_path = OUTPUT_DIR / "geometric_transform_grid.png"
    cv2.imwrite(str(output_path), tiled)

    cv2.imshow("Original | Translate | Rotate | Reflect", tiled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved geometric transform grid to: {output_path}")


if __name__ == "__main__":
    main()

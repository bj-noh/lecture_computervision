from pathlib import Path
import sys

import cv2


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    image_path = DATA_DIR / "cat.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    print(f"Loaded image shape: {image.shape}")
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = OUTPUT_DIR / "cat_copy.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"Saved copy to: {output_path}")


if __name__ == "__main__":
    main()

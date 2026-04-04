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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, dsize=(0, 0), fx=0.5, fy=0.5)

    cv2.imwrite(str(OUTPUT_DIR / "cat_gray.jpg"), gray)
    cv2.imwrite(str(OUTPUT_DIR / "cat_gray_small.jpg"), small)

    cv2.imshow("Original", image)
    cv2.imshow("Gray", gray)
    cv2.imshow("Gray Small", small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Original shape: {image.shape}")
    print(f"Gray shape: {gray.shape}")
    print(f"Resized shape: {small.shape}")


if __name__ == "__main__":
    main()

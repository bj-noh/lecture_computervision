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

    cv2.line(image, (20, 20), (200, 160), (255, 0, 0), 4)
    cv2.rectangle(image, (220, 40), (380, 180), (0, 255, 0), 3)
    cv2.circle(image, (130, 270), 55, (0, 0, 255), -1)
    cv2.putText(
        image,
        "OpenCV Lab",
        (210, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path = OUTPUT_DIR / "cat_annotated.jpg"
    cv2.imwrite(str(output_path), image)

    cv2.imshow("Draw Shapes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved annotated image to: {output_path}")


if __name__ == "__main__":
    main()

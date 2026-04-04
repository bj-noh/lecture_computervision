from pathlib import Path
import sys

import cv2


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def main() -> None:
    image_path = DATA_DIR / "cat.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    canvas = image.copy()

    def draw(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(canvas, (x, y), (x + 160, y + 120), (0, 0, 255), 2)
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.rectangle(canvas, (x, y), (x + 100, y + 100), (255, 0, 0), 2)
        cv2.imshow("Mouse Rectangle", canvas)

    cv2.namedWindow("Mouse Rectangle")
    cv2.setMouseCallback("Mouse Rectangle", draw)

    print("Left click: red rectangle | Right click: blue square | Press 'c' to clear | Press 'q' to quit")

    while True:
        cv2.imshow("Mouse Rectangle", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            canvas[:] = image
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

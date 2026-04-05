from pathlib import Path
import sys

import cv2


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    image_path = DATA_DIR / "apple.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=120,
        param1=150,
        param2=20,
        minRadius=20,
        maxRadius=140,
    )

    overlay = image.copy()
    circle_count = 0
    if circles is not None:
        for circle in circles[0]:
            x, y, radius = int(circle[0]), int(circle[1]), int(circle[2])
            cv2.circle(overlay, (x, y), radius, (255, 0, 0), 2)
            circle_count += 1

    output_path = OUTPUT_DIR / "hough_circles.png"
    cv2.imwrite(str(output_path), overlay)

    cv2.imshow("Hough Circles", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Detected circles: {circle_count}")
    print(f"Saved Hough circle result to: {output_path}")


if __name__ == "__main__":
    main()

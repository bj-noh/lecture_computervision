from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    image_path = DATA_DIR / "chess.png"
    image = cv2.imread(str(image_path))

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 130)

    overlay = image.copy()
    line_count = 0
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            line_count += 1

    merged = np.hstack([image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), overlay])
    output_path = OUTPUT_DIR / "hough_lines.png"
    cv2.imwrite(str(output_path), merged)

    cv2.imshow("Original | Edges | Hough Lines", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Detected lines: {line_count}")
    print(f"Saved Hough line result to: {output_path}")


if __name__ == "__main__":
    main()

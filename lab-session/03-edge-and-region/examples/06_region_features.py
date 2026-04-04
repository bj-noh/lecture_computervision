from pathlib import Path

import cv2
import numpy as np
from skimage import data


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    horse = data.horse()
    image = 255 - np.uint8(horse) * 255

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]

    moments = cv2.moments(contour)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    roundness = (4.0 * np.pi * area) / (perimeter * perimeter)
    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]

    preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    approx = cv2.approxPolyDP(contour, 8, True)
    hull = cv2.convexHull(contour).reshape(1, -1, 2)

    cv2.drawContours(preview, [contour], -1, (255, 0, 255), 2)
    cv2.drawContours(preview, [approx], -1, (0, 255, 0), 2)
    cv2.drawContours(preview, hull, -1, (0, 0, 255), 2)
    cv2.circle(preview, (int(center_x), int(center_y)), 4, (255, 255, 0), -1)

    output_path = OUTPUT_DIR / "region_features.png"
    cv2.imwrite(str(output_path), preview)

    cv2.imshow("Region Features", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Area: {area:.1f}")
    print(f"Center: ({center_x:.1f}, {center_y:.1f})")
    print(f"Perimeter: {perimeter:.1f}")
    print(f"Roundness: {roundness:.4f}")
    print(f"Saved region feature result to: {output_path}")


if __name__ == "__main__":
    main()

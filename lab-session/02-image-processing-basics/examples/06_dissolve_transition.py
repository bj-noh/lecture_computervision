from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    first_path = DATA_DIR / "dog1.jpg"
    second_path = DATA_DIR / "cat.jpg"

    first = cv2.imread(str(first_path))
    second = cv2.imread(str(second_path))

    if first is None or second is None:
        sys.exit("Could not open one of the dissolve images")

    second = cv2.resize(second, (first.shape[1], first.shape[0]))
    preview_frames = []

    for idx, alpha in enumerate(np.linspace(0.0, 1.0, 10)):
        frame = cv2.addWeighted(first, 1.0 - alpha, second, alpha, 0)
        cv2.imshow("Dissolve Transition", frame)
        cv2.waitKey(120)
        if idx in {0, 3, 6, 9}:
            preview_frames.append(cv2.resize(frame, (220, 160)))

    cv2.destroyAllWindows()

    preview = np.hstack(preview_frames)
    output_path = OUTPUT_DIR / "dissolve_preview.jpg"
    cv2.imwrite(str(output_path), preview)
    print(f"Saved dissolve preview to: {output_path}")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    if not hasattr(cv2, "SIFT_create"):
        sys.exit("This OpenCV build does not support SIFT.")

    image_path = DATA_DIR / "view.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        sys.exit(f"Could not open image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    preview = cv2.drawKeypoints(
        gray,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    descriptor_preview = descriptors[:3] if descriptors is not None and len(descriptors) >= 3 else descriptors
    output_path = OUTPUT_DIR / "sift_keypoints.png"
    cv2.imwrite(str(output_path), preview)

    cv2.imshow("SIFT Keypoints", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Keypoints: {len(keypoints)}")
    if descriptors is not None:
        print(f"Descriptor shape: {descriptors.shape}")
        print("First descriptors:")
        print(np.array2string(descriptor_preview, precision=2, suppress_small=True))
    print(f"Saved SIFT preview to: {output_path}")


if __name__ == "__main__":
    main()

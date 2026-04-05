from pathlib import Path
import sys
import time

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    if not hasattr(cv2, "SIFT_create"):
        sys.exit("This OpenCV build does not support SIFT.")

    path1 = DATA_DIR / "bus.jpg"
    path2 = DATA_DIR / "bus2.jpg"
    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))

    if img1 is None or img2 is None:
        sys.exit("Could not open one of the matching images")

    img1 = img1[10:668, 114:944]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        sys.exit("Could not compute descriptors for matching")

    start = time.time()
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_match = matcher.knnMatch(des1, des2, 2)

    good_match = []
    ratio_threshold = 0.7
    for nearest1, nearest2 in knn_match:
        if (nearest1.distance / nearest2.distance) < ratio_threshold:
            good_match.append(nearest1)

    elapsed = time.time() - start

    canvas = np.empty(
        (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
        dtype=np.uint8,
    )
    match_view = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_match,
        canvas,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    output_path = OUTPUT_DIR / "sift_flann_matches.png"
    cv2.imwrite(str(output_path), match_view)

    cv2.imshow("SIFT FLANN Matches", match_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Keypoints: {len(kp1)} vs {len(kp2)}")
    print(f"Good matches: {len(good_match)}")
    print(f"Matching time: {elapsed:.4f} sec")
    print(f"Saved match visualization to: {output_path}")


if __name__ == "__main__":
    main()

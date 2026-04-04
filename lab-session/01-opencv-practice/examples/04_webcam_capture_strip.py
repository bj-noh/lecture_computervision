from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    capture = cv2.VideoCapture(0)
    frames = []

    if not capture.isOpened():
        sys.exit("Could not open webcam")

    print("Press 'c' to capture a frame, 'q' to quit.")

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Failed to read frame")
            break

        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"Captured: {len(frames)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Webcam Capture Strip", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            frames.append(frame.copy())
            print(f"Captured frame {len(frames)}")
        elif key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

    if not frames:
        print("No frames captured")
        return

    max_frames = min(3, len(frames))
    resized = [cv2.resize(frame, (320, 240)) for frame in frames[:max_frames]]
    strip = np.hstack(resized)

    output_path = OUTPUT_DIR / "webcam_strip.jpg"
    cv2.imwrite(str(output_path), strip)
    cv2.imshow("Captured Strip", strip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved strip to: {output_path}")


if __name__ == "__main__":
    main()

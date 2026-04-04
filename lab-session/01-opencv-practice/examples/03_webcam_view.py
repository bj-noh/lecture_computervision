import sys

import cv2


def main() -> None:
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        sys.exit("Could not open webcam")

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Failed to read frame")
            break

        cv2.imshow("Webcam View", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import sys
import numpy as np
import cv2

def main():

    # path = '../data/pedestrians.avi'
    # cap = cv2.VideoCapture(path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Video open failed!')
        sys.exit()

    ret, back = cap.read()

    if not ret:
        print('Background image registration failed!')
        sys.exit()

    back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    back = cv2.GaussianBlur(back, (0, 0), 1.0)
    fback = back.astype(np.float32)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

        cv2.accumulateWeighted(gray, fback, 0.01)
        back = fback.astype(np.uint8)

        diff = cv2.absdiff(gray, back)
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff)

        for i in range(1, cnt):
            x, y, w, h, s = stats[i]

            if s < 100:
                continue

            cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        cv2.imshow('diff', diff)
        cv2.imshow('back', back)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

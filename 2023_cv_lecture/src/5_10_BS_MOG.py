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

    bs = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgmask = bs.apply(gray)
        back = bs.getBackgroundImage()

        cnt, _, stats, _ = cv2.connectedComponentsWithStats(fgmask)

        for i in range(1, cnt):
            x, y, w, h, s = stats[i]

            if s < 80:
                continue

            cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        cv2.imshow('back', back)
        cv2.imshow('fgmask', fgmask)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

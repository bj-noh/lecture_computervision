import cv2
import numpy as np
import sys

def main():
    path = '../data/pedestrians.avi' # Edit your image/video path

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        print('Video open failed!')
        sys.exit()


    # 배경 영상 등록
    ret, back = cap.read()
    back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

    while True:

        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Frame difference
            diff = cv2.absdiff(gray, back)

            _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            cv2.imshow('origin', frame)
            cv2.imshow('frame difference', diff)
            
            # Exit if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


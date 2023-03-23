import cv2
import numpy as np


def main():    
    path = '../data/view.jpg' # Edit your image path

    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp, des = sift.detectAndCompute(gray, None)

    print('keypoint:',len(kp), 'descriptor:', des.shape)
    print(des)

    gray = cv2.drawKeypoints(gray, kp, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('sift', gray)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

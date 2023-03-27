import cv2
import numpy as np
import time


def main():    
    path1 = '../data/bus.jpg' # Edit your image path
    path2 = '../data/bus2.jpg'

    
    img1 = cv2.imread(path1)[10:668, 114:944]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img1', img1)
    
    img2 = cv2.imread(path2)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img2', img2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    print("Number of keypoints: ", len(kp1), len(kp2))

    start = time.time()

    flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_match = flann_matcher.knnMatch(des1, des2, 2)

    T = 0.7
    good_match = []
    for nearest1, nearest2 in knn_match:
        if(nearest1.distance / nearest2.distance) < T:
            good_match.append(nearest1)

    print("Matching proc. time: ", time.time() - start)

    img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2[1].shape[1], 3), dtype = np.uint8)
    img_match = cv2.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('Good Matches', img_match)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

import cv2
import numpy as np


def main():
    path = '../data/dog1.jpg' # Edit your image path
    
    # Image Load
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 100, 200)

    # parameter를 통해 여러가지 근사 방법 제공
    contour, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    lcontour = []

    for i in range(len(contour)):
        if contour[i].shape[0] > 100: # 전체 길이가 100이상이면 하나의 line으로
            lcontour.append(contour[i])

    cv2.drawContours(image, lcontour, -1, (0, 255, 0), 3)
    
    cv2.imshow('Original with contours', image)
    cv2.imshow('Canny', canny)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

import cv2
import numpy as np


def main():
    path = '../data/apple.jpg' # Edit your image path
    
    # Image Load
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1 = 150, param2 = 20, minRadius = 30, maxRadius = 100)

    for i in circles[0]:
        cv2.circle(image, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 2)

    
    cv2.imshow('Hough circles', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

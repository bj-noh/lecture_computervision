import cv2
import numpy as np


def main():
    path = '../data/dog1.jpg' # Edit your image path
    
    # Image Load
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny1 = cv2.Canny(gray, 50, 150) # T_low = 50, T_high = 150
    canny2 = cv2.Canny(gray, 100, 200) # T_low = 100, T_high = 200

    cv2.imshow('Original with contours', image)
    cv2.imshow('Canny1', canny1)
    cv2.imshow('Canny2', canny2)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

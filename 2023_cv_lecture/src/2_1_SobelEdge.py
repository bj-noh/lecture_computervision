import cv2
import numpy as np


def main():
    path = '../data/dog1.jpg' # Edit your image path
    
    # Image Load
    image = cv2.imread(path, cv2.COLOR_BGR2GRAY)

    # Sobel 연산자 적용
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize = 3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize = 3)

    # 절댓값을 취해 양수 영상으로 변환
    sobel_x = cv2.convertScaleAbs(grad_x)
    sobel_y = cv2.convertScaleAbs(grad_y)

    edge_strength = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0) # Edge 강도 계산

    cv2.imshow('Original', image)
    cv2.imshow('sobel_x', sobel_x)
    cv2.imshow('sobel_y', sobel_y)
    cv2.imshow('edge_strength', edge_strength)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

import cv2
import numpy as np


def main():
    path = './data/view.png' # Edit your image path
    
    # Image Load
    image = cv2.imread(path)

    # 1) 가우시안 커널을 opencv 함수를 호출하여 활용
    g_kernel = cv2.getGaussianKernel(3, 0)
    g_blur1 = cv2.filter2D(image, -1, g_kernel*g_kernel.T)

    # 2) 가우시안 블러자체를 opencv 함수를 호출하여 활용 
    g_blur2 = cv2.GaussianBlur(image, (3, 3), 0)

    cv2.imshow('origin', image)
    cv2.imshow('g_blur1', g_blur1) 
    cv2.imshow('g_blur2', g_blur2) 
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

import cv2
import numpy as np


def main():
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0], 
                      [0, 0, 0, 1, 1, 1, 1, 0, 0, 0], 
                      [0, 0, 0, 1, 1, 1, 1, 1, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    

    uy = np.array([[-1, 0, 1]]).transpose()
    ux = np.array([[-1, 0, 1]])
    
    k = cv2.getGaussianKernel(3, 1)
    g = np.outer(k, k.transpose())

    dy = cv2.filter2D(image, cv2.CV_32F, uy)
    dx = cv2.filter2D(image, cv2.CV_32F, ux)

    dyy = dy * dy
    dxx = dx * dx
    dyx = dy * dx

    gdyy = cv2.filter2D(dyy, cv2.CV_32F, g)
    gdxx = cv2.filter2D(dxx, cv2.CV_32F, g)
    gdyx = cv2.filter2D(dyx, cv2.CV_32F, g)

    C = (gdyy * gdxx - gdyx * gdyx) - 0.04 * (gdyy + gdxx) * (gdyy + gdxx) # 특징 가능성 맵

    for j in range(1, C.shape[0] -1): # NMS
        for i in range(1, C.shape[1]-1):
            if C[j, i] > 0.1 and sum(sum(C[j, i] > C[j-1:j+2, i-1:i+2])) == 8:
                image[j, i] = 9 # 원본 영상에 9로 표시 
    
    np.set_printoptions(precision = 2)

    print(dy)
    print(dx)
    print(dyy)
    print(dxx)
    print(dyx)
    print(gdyy)
    print(gdxx)
    print(gdyx)
    print(C)
    print(image) # 원본 영상에 특징점을 9로 표시

    popping = np.zeros([160, 160], np.uint8) # 화소 확인 가능하도록 16배로 확대


    for j in range(0, popping.shape[0]):
        for i in range(0, popping.shape[1]):
            popping[j, i] = np.uint8((C[j//16, i//16]+0.06) * 700)

    cv2.imshow('Popping image display', popping)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

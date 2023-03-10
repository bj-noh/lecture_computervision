import cv2
import numpy as np


def main(option):
    path = './data/dog1.jpg' # Edit your image path

    if option == 'rgb':
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        cv2.imshow('rgb', src)

    elif option == 'rgba':
        src = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        cv2.imshow('bgra', src)
        cv2.imshow('alpha', src[:, :, 2])  # Only alpha channel

    elif option == 'gray1':
        src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('gray1', src)

    elif option == 'gray2':
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        gray2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray2', gray2)

    elif option == 'gray3':
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        src = src.astype(np.uint16) 
        b,g,r = cv2.split(src) 
        gray3 = ((b + g + r)/3).astype(np.uint8) 
        cv2.imshow('gray3', gray3)

    elif option == 'hsv1':
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        hsv1 = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        cv2.imshow('hsv1', hsv1)
    
    elif option == 'hsv2':
        red_bgr = np.array([[[0,0,255]]], dtype=np.uint8)   # 빨강 값만 갖는 픽셀
        green_bgr = np.array([[[0,255,0]]], dtype=np.uint8) # 초록 값만 갖는 픽셀
        blue_bgr = np.array([[[255,0,0]]], dtype=np.uint8)  # 파랑 값만 갖는 픽셀
        yellow_bgr = np.array([[[0,255,255]]], dtype=np.uint8) # 노랑 값만 갖는 픽셀

        red_hsv = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV)
        green_hsv = cv2.cvtColor(green_bgr, cv2.COLOR_BGR2HSV)
        blue_hsv = cv2.cvtColor(blue_bgr, cv2.COLOR_BGR2HSV)
        yellow_hsv = cv2.cvtColor(yellow_bgr, cv2.COLOR_BGR2HSV)
        
        print("red:",red_hsv)
        print("green:", green_hsv)
        print("blue", blue_hsv)
        print("yellow", yellow_hsv)

    elif option == 'ycbcr1':
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        ycbcr1 = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
        cv2.imshow('ycbcr1', ycbcr1)

    elif option == 'ycbcr2':
        dark = np.array([[[0,0,0]]], dtype=np.uint8)        # 3 채널 모두 0인 가장 어두운 픽셀
        middle = np.array([[[127,127,127]]], dtype=np.uint8) # 3 채널 모두 127인 중간 밝기 픽셀
        bright = np.array([[[255,255,255]]], dtype=np.uint8) # 3 채널 모두 255인 가장 밝은 픽셀

        dark_yuv = cv2.cvtColor(dark, cv2.COLOR_BGR2YUV)
        middle_yuv = cv2.cvtColor(middle, cv2.COLOR_BGR2YUV)
        bright_yuv = cv2.cvtColor(bright, cv2.COLOR_BGR2YUV)

        print("dark:",dark_yuv)
        print("middle:", middle_yuv)
        print("bright", bright_yuv)

    else:
        print("Please enter the correct option")

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('rgba')


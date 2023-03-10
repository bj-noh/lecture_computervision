import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    path = './Data/view.png' # Edit your image path
    
    # Image Load
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 1) 가우시안 커널을 opencv 함수를 호출하여 활용
    g_kernel = cv2.getGaussianKernel(3, 0)
    g_blur1 = cv2.filter2D(image, -1, g_kernel*g_kernel.T)

    # 2) 가우시안 블러자체를 opencv 함수를 호출하여 활용 
    g_blur2 = cv2.GaussianBlur(image, (3, 3), 0)

    # View
    plt.subplot(1,3,1)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(image, cmap='gray')

    plt.subplot(1,3,2)
    plt.title('g_blur2')
    plt.axis('off')
    plt.imshow(g_blur1, cmap='gray')

    plt.subplot(1,3,3)
    plt.title('g_blur2')
    plt.axis('off')
    plt.imshow(g_blur2, cmap='gray')

    plt.show()

if __name__ == '__main__':
    main()

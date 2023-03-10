import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1 / gamma
    output = np.uint8(((image / 255) ** inv_gamma) * 255)
    return output


def main():
    path = './Data/dog1.jpg' # Edit your image path
    # Gamma Correction Equation
    
    # Image Load
    image = cv2.imread(path)

    # Grayscale
    gray_image = np.float32(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

    output1 = gamma_correction(gray_image, gamma=0.5) # gamma value = 0.5 
    output2 = gamma_correction(gray_image, gamma=2.0) # gamma value = 2.0

    # View
    plt.subplot(1,3,1)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(gray_image, cmap='gray')

    plt.subplot(1,3,2)
    plt.title('Gamma = 0.5')
    plt.axis('off')
    plt.imshow(output1, cmap='gray')

    plt.subplot(1,3,3)
    plt.title('Gamma = 2.2')
    plt.axis('off')
    plt.imshow(output2, cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()

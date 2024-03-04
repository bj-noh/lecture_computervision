import cv2
import numpy as np

def resize_image_to_match(image1, image2):
    height, width = image1.shape[:2]
    resized_image2 = cv2.resize(image2, (width, height))
    return resized_image2

def dissolve_effect(image1, image2, steps=30):
    image2 = resize_image_to_match(image1, image2)  
    for alpha in np.linspace(0, 1, steps):
        beta = 1.0 - alpha
        dst = cv2.addWeighted(image1, alpha, image2, beta, 0)
        cv2.imshow('Dissolve Transition', dst)
        cv2.waitKey(100)  
    cv2.destroyAllWindows()

'''
'''
image1 = cv2.imread('path_to_your_first_image.jpg')
image2 = cv2.imread('path_to_your_second_image.jpg')

dissolve_effect(image1, image2, steps=30)

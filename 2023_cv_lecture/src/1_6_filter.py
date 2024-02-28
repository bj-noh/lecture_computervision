import cv2
import numpy as np


def main():
    path = './data/view.png' # Edit your image path
    
    # Image Load
    image = cv2.imread(path)


    kernel = np.ones((5, 5))/5**2 # 5x5 평균 filter kernel 생성  생성
    '''
    kernel = np.array([[0.0, 0.04, 0.04, 0.04, 0.04],
                      [0.04, 0.04, 0.04, 0.04, 0.04],
                      [0.04, 0.04, 0.04, 0.04, 0.04],
                      [0.04, 0.04, 0.04, 0.04, 0.04],
                      [0.04, 0.04, 0.04, 0.04, 0.04]])
    '''

    blurred = cv2.filter2D(image, -1, kernel)
        
    cv2.imshow('origin', image)
    cv2.imshow('avrg blur', blurred) 
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

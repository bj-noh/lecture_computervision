import cv2
import numpy as np

def main():
    path = './data/dog1.jpg' # Edit your image path
    src = cv2.imread(path)
    rows, cols = src.shape[:2]
    mask = np.zeros((rows+2, cols+2), np.uint8)
    newVal = (150, 25, 150) # Change your color
    loDiff, upDiff = (10,10,10), (10,10,10)


    # 마우스 이벤트 처리 함수
    def onMouse(event, x, y, flags, param):
        # global mask, src
        if event == cv2.EVENT_LBUTTONDOWN:
            seed = (x,y)
            retval = cv2.floodFill(src, m
            ask, seed, newVal, loDiff, upDiff)
        
            cv2.imshow('src', src)

    # 화면 출력
    cv2.imshow('src', src)
    cv2.setMouseCallback('src', onMouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main()

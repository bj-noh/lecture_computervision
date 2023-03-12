import cv2
import numpy as np


def main():
    path = '../data/chess.png' # Edit your image path
    
    # Image Load
    image = cv2.imread(path)
    image2 = image.copy()
    h, w = image.shape[:2]
   
    # 그레이 스케일 변환 및 엣지 검출
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # 허프 선 검출, 직선으로 판단할 최소한의 점은 130개로 지정
    lines = cv2.HoughLines(edges, 1, np.pi/180, 130)

    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0+1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 -1000*(a))

            cv2.line(image2, (x1, y1), (x2, y2),(0, 0, 255), 2)

    #결과 출력    
    merged = np.hstack((image, image2))
    cv2.imshow('Hough line', merged)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

import cv2
import sys

img = cv2.imread('../data/apple.jpg')    

if img is None:
    sys.exit('파일을 찾을 수 없습니다')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_small = cv2.resize(gray, dsize=(0,0), fx=0.5, fy=0.5) # 반으로 축소

cv2.imwrite('../data/apple_gray.jpg', gray)
cv2.imwrite('../data/apple_gray_small.jpg', gray_small)

cv2.imshow('Image Window', img)
cv2.imshow('Gray Image', gray)
cv2.imshow('Gray Image Small', gray_small)
cv2.waitKey(0) # 키보드 입력을 기다린다.
cv2.destroyAllWindows()
    
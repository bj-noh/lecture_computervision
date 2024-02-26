import cv2
import sys

# 이미지 읽기
img = cv2.imread('../data/apple.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다')

# 이미지 보기
cv2.imshow('Image Window', img)
cv2.waitKey(0) # 키보드 입력을 기다린다.
cv2.destroyAllWindows()

cv2.imwrite('../data/apple_copy.jpg', img)

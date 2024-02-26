import cv2
import sys

# 이미지 읽기
img = cv2.imread('../data/apple.jpg')
if img is None:
    print("Error: Image not found.")


# 이미지에 선 그리기
cv2.line(img, (0, 0), (150, 150), (255, 0, 0), 5) # 파란색 선

# 이미지에 사각형 그리기
cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 3) # 초록색 사각형

# 이미지에 원 그리기
cv2.circle(img, (250, 250), 50, (0, 0, 255), -1) # 빨간색 원

# 이미지에 텍스트 쓰기
cv2.putText(img, 'Apple', (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 그려진 이미지 보기
cv2.imshow('Drawn Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
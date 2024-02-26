import cv2
import sys

# 이미지 읽기
img = cv2.imread('../data/apple.jpg')
if img is None:
    print("Error: Image not found.")

def draw(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭
        cv2.rectangle(img, (x, y), (x+200, y+200), (0, 0, 255), 2)
        
    elif event == cv2.EVENT_RBUTTONDOWN: # 마우스 오른쪽 버튼 클릭
        cv2.rectangle(img, (x, y), (x+100, y+100), (255, 0, 0), 2)
        
    cv2.imshow('Drawing', img)
    
    
cv2.namedWindow('Drawing')
cv2.imshow('Drawing', img)

cv2.setMouseCallback('Drawing', draw)

while True:
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
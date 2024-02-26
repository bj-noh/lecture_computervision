import cv2
import numpy as np
import sys

cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    sys.exit('카메라 연결 실패')
    
frames = []
while True:
    ret, frame = cap.read()
    
    if not ret:
        print('프레임 획득 실패')
        break
    
    cv2.imshow('Video Display', frame)
    
    key = cv2.waitKey(1) # 1ms 동안 키보드 입력 기다림
    
    if key == ord('c'):
        frames.append(frame)
        
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
   
if len(frames)>0:
    imgs = frames[0]
    
    for i in range(1, min(3, len(frames))):
        imgs = np.hstack((imgs, frames[i]))
        
    cv2.imshow('Collected Images', imgs)
    
    cv2.waitKey()
    cv2.destroyAllWindows()

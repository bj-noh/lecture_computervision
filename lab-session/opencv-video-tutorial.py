import cv2
import sys

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    sys.exit("Could not open video stream")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
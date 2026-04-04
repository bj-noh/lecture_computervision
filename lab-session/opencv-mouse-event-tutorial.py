import cv2
import sys

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

img = cv2.imread("cat.jpg")

if img is None:
    sys.exit("Could not open or find the image")

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_circle)

while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
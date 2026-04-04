import cv2
import sys

im = cv2.imread("cat.jpg")

print(im.shape)

if im is None:
    sys.exit("Could not open or find the image")

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
resized_gray = cv2.resize(gray, dsize=(0, 0), fx=2.5, fy=2.5)

cv2.imshow("Display window", resized_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(resized_gray.shape)

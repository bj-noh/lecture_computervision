import cv2

def main():
    path = '../Data/dog1.jpg' # Edit your image path
    src = cv2.imread(path)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("original", src)
    cv2.imshow("binary", dst)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

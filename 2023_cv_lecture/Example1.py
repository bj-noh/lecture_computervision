import cv2

def main():
    path = '../Data/dog1.jpg' # Edit your image path
    img = cv2.imread(path)
    # height, width = img.shape[:2]

    cv2.imshow('dog_image', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

import cv2

def main():
    path = './Data/dog1.jpg' # Edit your image path

    img = cv2.imread(path)
    print(type(img))
    print(type(img[0, 0, 0]))


if __name__ == '__main__':
    main()

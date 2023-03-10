import cv2

def main():
    path = './Data/dog1.jpg' # Edit your image path
    img = cv2.imread(path)
    # gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flip = cv2.flip(img, 0)

    # Process    
    cv2.imshow('dog_image', img)
    cv2.imshow('Gray', gray)
    cv2.imshow('Flip', flip)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()

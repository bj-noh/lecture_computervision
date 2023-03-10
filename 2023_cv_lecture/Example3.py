import cv2

def main():
    path = './Data/dog1.jpg' # Edit your image path
    img = cv2.imread(path)
    
    # Process    
    shape_rs = img.shape
    print(shape_rs) 

    height, width = img.shape[:2]
    print(height, width)
    # cv2.imshow('dog_image', img)
    # cv2.waitKey(0)
if __name__ == '__main__':
    main()

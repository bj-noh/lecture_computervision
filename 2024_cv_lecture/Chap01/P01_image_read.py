import cv2
import sys

def image_read(image_file):
    # 이미지 읽기
    img = cv2.imread(image_file)
    
    if img is None:
        sys.exit('파일을 찾을 수 없습니다')

    # 이미지 보기
    cv2.imshow('Image Window', img)
    cv2.waitKey(0) # 키보드 입력을 기다린다.
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_read('../data/apple.jpg')
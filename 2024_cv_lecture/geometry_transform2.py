import cv2
import numpy as np

# 이미지 로드
image_path = "./data/dog1.jpg"  # 이미지 경로를 적절히 수정하세요.
image = cv2.imread(image_path)

# 이미지 이동 변환
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

# 이미지 회전 변환
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# 이미지 스케일링 변환
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# 축에 대한 반사 변환
def flip(image, axis=0):
    # axis=0: x축, axis=1: y축
    flipped = cv2.flip(image, axis)
    return flipped

# 이미지 합치기
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

# 변환 적용
translated_image = translate(image, x=100, y=50)
rotated_image = rotate(image, angle=45)
resized_image = resize(image, width=200)
flipped_image = flip(image, axis=1)  # y축에 대한 반사

# 2x2 이미지로 합치기
final_image = concat_tile([[translated_image, resized_image], [rotated_image, flipped_image]])

# 최종 결과 표시
cv2.imshow("Result", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

# 이미지 로드
image_path = "./data/dog1.jpg"  # 이미지 경로를 적절히 수정하세요.
image = cv2.imread(image_path)

# 이미지의 높이와 너비
(h, w) = image.shape[:2]

# 이동 변환 행렬
T = np.float32([[1, 0, 100], 
                [0, 1, 50], 
                [0, 0, 1]])

# 회전 변환 행렬 (중심에서 45도 회전)
angle = np.radians(45)  # 각도를 라디안으로 변환
cos = np.cos(angle)
sin = np.sin(angle)
cx, cy = w / 2, h / 2  # 이미지의 중심
R = np.float32([[cos, -sin, cx * (1 - cos) + cy * sin], 
                [sin, cos, cy * (1 - cos) - cx * sin], 
                [0, 0, 1]])

# 스케일링 변환 행렬
S = np.float32([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])

# y축에 대한 반사 변환 행렬
Ry = np.float32([[-1, 0, image.shape[1]], [0, 1, 0], [0, 0, 1]])

# 변환 적용
translated_image = cv2.warpPerspective(image, T, (image.shape[1], image.shape[0]))
rotated_image = cv2.warpPerspective(image, R, (image.shape[1], image.shape[0]))
scaled_image = cv2.warpPerspective(image, S, (image.shape[1], image.shape[0]))
flipped_image = cv2.warpPerspective(image, Ry, (image.shape[1], image.shape[0]))

# 이미지 합치기
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

final_image = concat_tile([[image, scaled_image], [rotated_image, flipped_image]])
# 결과 이미지 표시
cv2.imshow("Result", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
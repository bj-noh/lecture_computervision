import cv2
import numpy as np

# 이미지를 로드합니다.
image = cv2.imread('./data/dog1.jpg', cv2.IMREAD_GRAYSCALE)  # 'your_image_path.jpg'에 이미지 경로를 넣으세요.

# Gaussian Blur를 적용합니다.
blurred_image = cv2.GaussianBlur(image, (3, 3), 0.1)

# Laplacian을 적용하여 LoG를 구현합니다.
log_image = cv2.Laplacian(blurred_image, cv2.CV_64F)

# 결과 이미지를 보기 좋게 변환합니다.
log_image = np.uint8(np.absolute(log_image))

# 원본 이미지와 LoG 필터가 적용된 이미지를 화면에 표시합니다.
cv2.imshow('Original Image', image)
cv2.imshow('Laplacian of Gaussian Image', log_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

# 이미지 유사성 평가 함수
def evaluate_image_similarity(descriptor1, descriptor2, threshold=10):
    flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_match = flann_matcher.knnMatch(descriptor1, descriptor2, 2)

    T = 0.7

    good_match = []
    for nearest1, nearest2 in knn_match:
        if (nearest1.distance / nearest2.distance) < T:
            good_match.append(nearest1)

    # 유사성 판단
    print(len(good_match))
    print(threshold)
    return len(good_match) > threshold

# 이미지 선택 시 효과 적용 함수
def apply_effect_on_click(event, x, y, flags, param):
    global selected_images, images, descriptors, comparison_started

    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭한 위치에 있는 이미지 식별
        img_index = (y // img_height) * 5 + (x // img_width)
        if img_index < len(images):
            # 이미지 선택 처리
            if img_index not in selected_images:
                selected_images.append(img_index)
                # 선택 효과 적용 (이미지 반전)
                images[img_index] = cv2.bitwise_not(images[img_index])
            else:
                # 이미 선택된 이미지를 다시 클릭한 경우, 선택 해제
                selected_images.remove(img_index)
                # 원본 이미지로 복원
                images[img_index] = cv2.imread(image_paths[img_index], cv2.IMREAD_COLOR)

            # 선택된 이미지가 2개인 경우, 유사성 평가 시작
            if len(selected_images) == 2:
                comparison_started = True

# 이미지 로딩
image_paths = ['../data/apple.jpg', '../data/apple.jpg', '../data/apple_copy.jpg', '../data/apple.jpg', '../data/object.jpg', 
               './images/apple3.webp', './images/apple4.webp', '../data/bus.jpg', '../data/bus2.jpg', '../data/mon.png']  # 이미지 경로 리스트
images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
print(image_paths)

# SIFT로 이미지에서 기술자 추출
sift = cv2.SIFT_create()
descriptors = [sift.detectAndCompute(img, None)[1] for img in images]

# 이미지 크기 조정 (동일한 크기로)
img_width, img_height = 200, 200
images = [cv2.resize(img, (img_width, img_height)) for img in images]

# 윈도우 및 마우스 콜백 함수 설정
cv2.namedWindow('Images')
cv2.setMouseCallback('Images', apply_effect_on_click)

# 선택된 이미지와 비교 시작 플래그 초기화
selected_images = []
comparison_started = False

# 이미지를 2x5 그리드로 표시
grid_img = np.zeros((2*img_height, 5*img_width, 3), dtype=np.uint8)
for i, img in enumerate(images):
    row = i // 5
    col = i % 5
    grid_img[row*img_height:(row+1)*img_height, col*img_width:(col+1)*img_width] = img

while True:
    cv2.imshow('Images', grid_img)
    cv2.waitKey(100)
    if comparison_started:
        # 두 이미지의 유사성 평가
        if evaluate_image_similarity(descriptors[selected_images[0]], descriptors[selected_images[1]]):
            print("두 이미지는 유사합니다.")
            # 유사한 이미지를 blank로 처리
            for idx in selected_images:
                row = idx // 5
                col = idx % 5
                grid_img[row*img_height:(row+1)*img_height, col*img_width:(col+1)*img_width] = 0
        else:
            print("두 이미지는 유사하지 않습니다.")
        selected_images = []
        comparison_started = False

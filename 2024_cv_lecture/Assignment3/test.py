import cv2
import numpy as np

def evaluate_image_similarity(img1_path, img2_path, threshold=10):
    # 이미지를 읽어옵니다.
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # SIFT 특징 검출기를 생성합니다.
    sift = cv2.SIFT_create()

    # 각 이미지에서 특징점과 기술자를 검출합니다.
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_match = flann_matcher.knnMatch(descriptors1, descriptors2, 2)

    T = 0.7

    good_match = []
    for nearest1, nearest2 in knn_match:
        if (nearest1.distance / nearest2.distance) < T:
            good_match.append(nearest1)

    # 좋은 매치의 수를 기반으로 유사성을 평가합니다.
    print(len(good_match))
    print(threshold)
    if len(good_match) > threshold:
        print("유사한 이미지입니다.")
    else:
        print("유사하지 않은 이미지입니다.")

# 이미지 경로
img1_path = './images/apple3.webp'
img2_path = './images/apple4.webp'

evaluate_image_similarity(img1_path, img2_path)

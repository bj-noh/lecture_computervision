import cv2
import numpy as np
from tkinter import messagebox
import tkinter as tk

# Tkinter 메시지 박스를 위한 루트 윈도우 초기화 (하지만 메시지 박스만 사용)
root = tk.Tk()
root.withdraw()  # 메인 윈도우를 숨깁니다.

def evaluate_image_similarity(descriptor1, descriptor2, threshold=10):
    flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_match = flann_matcher.knnMatch(descriptor1, descriptor2, 2)

    T = 0.7

    good_match = []
    for nearest1, nearest2 in knn_match:
        if (nearest1.distance / nearest2.distance) < T:
            good_match.append(nearest1)

    print(len(good_match))
    print(threshold)

    return len(good_match) >= threshold

def apply_red_mask(image):
    # 이미지 위에 붉은색 마스크 적용
    red_mask = np.zeros_like(image)
    red_mask[:, :] = [0, 0, 255]  # BGR
    masked_image = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0)
    return masked_image

def on_mouse_click(event, x, y, flags, param):
    global selected_images, images, descriptors, comparison_started

    if event == cv2.EVENT_LBUTTONDOWN:
        img_index = (y // img_height) * 5 + (x // img_width)
        if img_index < len(images):
            if img_index not in selected_images:
                selected_images.append(img_index)
                images[img_index] = apply_red_mask(images[img_index])
            else:
                selected_images.remove(img_index)
                images[img_index] = cv2.imread(image_paths[img_index], cv2.IMREAD_COLOR)
                images[img_index] = cv2.resize(images[img_index], (img_width, img_height))

            if len(selected_images) == 2:
                comparison_started = True

image_paths = ['./images/apple2.webp', './images/bus2.webp', './images/cycle1.webp', './images/chess2.webp', './images/city2.webp',
               './images/bus1.webp', './images/city1.webp', './images/cycle2.webp', './images/apple1.webp', './images/chess1.webp']  # 이미지 경로 리스트

images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]

sift = cv2.SIFT_create()
descriptors = [sift.detectAndCompute(img, None)[1] for img in images]

img_width, img_height = 200, 200
images = [cv2.resize(img, (img_width, img_height)) for img in images]

cv2.namedWindow('Images')
cv2.setMouseCallback('Images', on_mouse_click)

selected_images = []
comparison_started = False

while True:
    grid_img = np.zeros((2*img_height, 5*img_width, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        row = i // 5
        col = i % 5
        grid_img[row*img_height:(row+1)*img_height, col*img_width:(col+1)*img_width] = img

    cv2.imshow('Images', grid_img)
    key = cv2.waitKey(1) & 0xFF
    if comparison_started:
        is_similar = evaluate_image_similarity(descriptors[selected_images[0]], descriptors[selected_images[1]])
        if is_similar:
            messagebox.showinfo("결과", "두 이미지는 유사합니다.")
            for idx in selected_images:
                images[idx] = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        else:
            messagebox.showinfo("결과", "두 이미지는 유사하지 않습니다.")
        selected_images = []
        comparison_started = False

    if key == 27:  # ESC 키
        break

cv2.destroyAllWindows()

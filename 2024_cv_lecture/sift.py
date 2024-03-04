import cv2 as cv

img = cv.imread('./data/bus.jpg')  # 영상 읽기
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

print(des)
print(des[0])
print(len(des))
print(len(des[0]))
print(len(des[1]))
print(len(des[50]))
first_keypoint = list(kp)[0]
# pt, size, angle, response, octave, class_id
print(f"Position: {first_keypoint.pt}")
print(f"Size: {first_keypoint.size}")
print(f"Angle: {first_keypoint.angle}")
print(f"Response: {first_keypoint.response}")
print(f"Octave: {first_keypoint.octave}")
print(f"Class ID: {first_keypoint.class_id}")


gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imshow('sift', gray)

# k = cv.waitKey(0)
# cv.destroyAllWindows()

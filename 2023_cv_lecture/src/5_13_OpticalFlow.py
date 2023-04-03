import cv2
import numpy as np

# read the first frame of the video
video_path = '../data/pedestrians.avi'
cap = cv2.VideoCapture(video_path)
ret, frame1 = cap.read()

# convert the first frame to grayscale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# create a mask for drawing the flow vectors
mask = np.zeros_like(frame1)

while True:
    # read the next frame of the video
    ret, frame2 = cap.read()

    if not ret:
        break

    # convert the current frame to grayscale
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # draw the flow vectors on the mask
    step_size = 16
    y, x = np.mgrid[step_size/2:frame1.shape[0]:step_size, step_size/2:frame1.shape[1]:step_size].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(mask, lines, 0, (0, 255, 0))

    # overlay the flow vectors on the current frame
    img = cv2.add(frame2, mask)

    # display the resulting frame
    cv2.imshow("Optical Flow", img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # update the previous frame
    prvs = next

# release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

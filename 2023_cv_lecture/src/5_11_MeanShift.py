import cv2
import numpy as np

def main():
    # Define the video capture device (in this case, the default webcam)
    cap = cv2.VideoCapture(0)

    # Define the initial region of interest (ROI)
    ret, frame = cap.read()
    r, h, c, w = 300, 200, 100, 200
    track_window = (c, r, w, h)

    # Convert the ROI to the HSV color space
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Create a histogram of the ROI in the HSV color space
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Set the termination criteria for the MeanShift algorithm
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        # Read a frame from the video capture device
        ret, frame = cap.read()

        if ret:
            # Convert the frame to the HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Perform back projection of the histogram onto the frame
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # Apply the MeanShift algorithm to find the new position of the ROI
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # Draw a rectangle around the new position of the ROI
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

            # Display the frame with the new ROI position
            cv2.imshow('img2', img2)

            # Exit if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

import cv2
import sys

# Load a video file
video_path = "../data/tracking1.mp4"
video = cv2.VideoCapture(video_path)
# video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open video.")
    sys.exit()

# Read the first frame
ok, frame = video.read()
if not ok:
    print("Error: Could not read the video file.")
    sys.exit()

# Define an initial bounding box
bbox = (500, 330, 300, 220)  # Example: (x, y, width, height)

# Create the KCF tracker
tracker = cv2.TrackerKCF_create()

# Initialize the tracker with the first frame and the bounding box
ok = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Update the tracker
    ok, bbox = tracker.update(frame)

    # Draw the bounding box if the tracking was successful
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
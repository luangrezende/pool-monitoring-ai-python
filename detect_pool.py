import cv2
import numpy as np

# Set the video path
video_path = "pool.mp4"

# Define the 4 pool coordinates (x, y) - top-left, top-right, bottom-right, bottom-left
pool_coordinates = np.array([
    [480, 350],  # Top-left corner
    [1440, 560],  # Top-right corner
    [1250, 980],  # Bottom-right corner
    [60, 450]    # Bottom-left corner
], dtype=np.int32)

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file!")
    exit()

while True:
    # Read the frame
    ret, frame = cap.read()
    
    # If unable to read frame (end of video), restart
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # Draw the rectangle (polygon) around the pool
    cv2.polylines(frame, [pool_coordinates], True, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Pool Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

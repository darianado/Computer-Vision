""" 
Parking Occupancy Detection
----------------------------
Student Names: Dariana Dorin - 2163317
               Djibril Coulybaly - 2162985
Description of File: Using the masking image and the video file, the program will 
                     apply the bounding boxes i.e. parking spaces to each frame of 
                     the video and process it accordingly 
"""

# Importing essential libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Adding the correct file paths for the mask and video
mask_image_path = r"mask.png"
video_path = r"video.mp4"

# Try locating the mask image
if not os.path.exists(mask_image_path):
    print(f"Mask image file not found at path: {mask_image_path}")

# Try locating the video
if not os.path.exists(video_path):
    print(f"Video file not found at path: {video_path}")

# Try reading the video file
video_capture = cv2.VideoCapture(video_path)
if not video_capture.isOpened():
    print(f"Failed to open video file: {video_path}")
else:
    print("Video file opened successfully.")

# Try reading the mask image file
mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print(f"Failed to read mask image file: {mask_image_path}")
else:
    print("Mask image file read successfully.")


# Function to draw the bounding boxes i.e. parking spaces
def draw_bounding_boxes(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    contours, _ = cv2.findContours(
        segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


# Process and show each frame of the video
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to read frame from video or end of video reached.")
        break

    # Process the frame
    processed_frame = draw_bounding_boxes(frame, mask)

    # Display the frame
    cv2.imshow("Processed Video", processed_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord("q"):  # Increase delay to 25 ms
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()

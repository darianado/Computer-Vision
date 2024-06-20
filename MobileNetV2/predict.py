""" 
Parking Occupancy Detection
----------------------------
Student Names: Dariana Dorin - 2163317
               Djibril Coulybaly - 2162985
Description of File: Using the trained MobileNet2 model, the program will 
                     apply the bounding boxes i.e. parking spaces to each frame of 
                     the video, and predict if the parking space is occupied or not. 
                     The occupancy will be highlighted (Green = Free, Red = Occupied)
                     and a count score will be monitored
"""

# Importing essential libraries
import os
import cv2
import numpy as np
import tensorflow as tf


# Function to resise the desired area for the bounding box to the target size that is specific to the training model used
def preprocess_for_prediction(desired_area, target_size=(75, 32)):
    desired_area_resized = cv2.resize(desired_area, target_size)
    desired_area_normalized = desired_area_resized / 255.0
    desired_area_expanded = np.expand_dims(desired_area_normalized, axis=0)
    return desired_area_expanded


# Class to represent a bounding box, with the dimensions and border colour initialised
class BoundingBox:
    def __init__(self, x, y, w, h, class_id):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.class_id = class_id

    def draw(self, frame):
        color = (0, 255, 0) if self.class_id == 0 else (0, 0, 255)
        cv2.rectangle(
            frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color, 2
        )


# Function to draw the bounding boxes on an input frame and predict the occupancy of each box
def draw_bounding_boxes_and_predict(frame, mask, model, bounding_boxes):
    # Turn frame into black and white and mask the different regions/features of the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    contours, _ = cv2.findContours(
        segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Finding contours of parking space and add the bounding box to it
    desired_areas = []
    new_bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        desired_area = frame[y : y + h, x : x + w]
        desired_area_preprocessed = preprocess_for_prediction(desired_area)
        desired_areas.append(desired_area_preprocessed)
        new_bounding_boxes.append(BoundingBox(x, y, w, h, 0))

    # Predict occupancy on parking space using trained model
    if desired_areas:
        desired_areas_batch = np.vstack(desired_areas)
        predictions = model.predict(desired_areas_batch)
        predicted_classes = np.argmax(predictions, axis=1)

        for i, bounding_box in enumerate(new_bounding_boxes):
            bounding_box.class_id = predicted_classes[i]

    # Update and draw bounding boxes
    bounding_boxes.clear()
    bounding_boxes.extend(new_bounding_boxes)
    for bounding_box in bounding_boxes:
        bounding_box.draw(frame)

    return frame


# Function to display number of non-occupied parking spaces
def draw_empty_spots_counter(frame, bounding_boxes):
    total_spots = len(bounding_boxes)
    empty_spots = sum(bbox.class_id == 0 for bbox in bounding_boxes)
    text = f"Empty Spots: {empty_spots}/{total_spots}"

    # Counter Dimentions
    box_x, box_y = 10, 10
    box_w, box_h = 200, 50

    # Draw counter box and apply text
    cv2.rectangle(
        frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), -1
    )
    cv2.putText(
        frame,
        text,
        (box_x + 10, box_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )

    return frame


# Paths to the trained model, masked image and video
mask_image_path = os.path.join("..","mask.png")
video_path = os.path.join("..","video.mp4")
model_dir = os.path.join("..","models")
model_filename = "model_mobilenetv2.keras"
model_path = os.path.join(model_dir, model_filename)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Check to see if paths exist
if not os.path.exists(mask_image_path):
    print(f"Mask image file not found at path: {mask_image_path}")
if not os.path.exists(video_path):
    print(f"Video file not found at path: {video_path}")

# Open the video file
video_capture = cv2.VideoCapture(video_path)
if not video_capture.isOpened():
    print(f"Failed to open video file: {video_path}")
else:
    print("Video file opened successfully.")

# Reading the mask image
mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print(f"Failed to read mask image file: {mask_image_path}")
else:
    print("Mask image file read successfully.")


bounding_boxes = []
# Process video frames one by one
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab a frame")
        break

    frame_with_predictions = draw_bounding_boxes_and_predict(
        frame, mask, model, bounding_boxes
    )
    frame_with_predictions = draw_empty_spots_counter(
        frame_with_predictions, bounding_boxes
    )
    # Display Results
    cv2.imshow("Video with Parking Slot Occupancy", frame_with_predictions)
    key = cv2.waitKey(0)  # Wait for Enter key press to process the next frame
    if key & 0xFF == ord("q"):
        break

# Terminate OpenCV resources when everything is done
video_capture.release()
cv2.destroyAllWindows()

""" 
Parking Occupancy Detection
----------------------------
Student Names:          Dariana Dorin - 2163317
                        Djibril Coulybaly - 2162985

Description of File:    The program will overlay the video frame on a black background, allowing us to draw the 
                        required bounding boxes i.e. parking spaces and create our mask
"""

# Importing essential libraries
import cv2
import numpy as np


# Function to draw the rectangle bounding box of a fixed dimension on a black overlay of the video frame
def create_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, mask, w, h

    # Update the initial position
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # Display the rectangle while moving the mouse, overlay the mask with existing rectangles and draw the new rectangle on the temporary image
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img.copy()
            temp_img = cv2.addWeighted(temp_img, 0.8, mask, 0.2, 0)
            cv2.rectangle(
                temp_img,
                (ix - w // 2, iy - h // 2),
                (ix + w // 2, iy + h // 2),
                (255, 255, 255),
                -1,
            )
            cv2.imshow("Create Mask", temp_img)

    # Draw the rectangle permanently on the mask image and update the image display
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(
            mask,
            (ix - w // 2, iy - h // 2),
            (ix + w // 2, iy + h // 2),
            (255, 255, 255),
            -1,
        )
        display_img = img.copy()
        display_img = cv2.addWeighted(display_img, 0.8, mask, 0.2, 0)
        cv2.imshow("Create Mask", display_img)


# Dimensions for the rectangle
w, h = 70, 27

# Open the video file
video_path = r"video.mp4"
video_capture = cv2.VideoCapture(video_path)
if not video_capture.isOpened():
    print(f"Failed to open video file: {video_path}")
else:
    print("Video file opened successfully.")

    ret, img = video_capture.read()

    if not ret:
        print("Failed to grab a frame")
    else:
        # Create black background for mask
        mask = np.zeros_like(img)
        mask.fill(0)

        cv2.namedWindow("Create Mask")
        cv2.setMouseCallback("Create Mask", create_rectangle)

        display_img = img.copy()
        display_img = cv2.addWeighted(display_img, 0.8, mask, 0.2, 0)
        cv2.imshow("Create Mask", display_img)

        # Press the ESC key to complete the drawing of the mask
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        # Terminate OpenCV resources when everything is done
        cv2.destroyAllWindows()
        video_capture.release()

        # Save the final mask as a PNG image
        cv2.imwrite("mask.png", mask)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Previous Attempt 1 - Draw the rectangle manually without a pre defined dimension  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # Importing essential libraries
# import cv2
# import numpy as np


# # Function to draw the rectangle bounding box of a fixed dimension on a black overlay of the video frame
# def create_rectangle(event, x, y, flags, param):
#     global ix, iy, drawing, img, mask

#     # Update the initial position
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y

#     # Display the rectangle while moving the mouse, overlay the mask with existing rectangles and draw the new rectangle on the temporary image
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             mask_copy = mask.copy()
#             cv2.rectangle(mask_copy, (ix, iy), (x, y), (255, 255, 255), -1)
#             display_img = cv2.addWeighted(img, 0.8, mask_copy, 0.2, 0)
#             cv2.imshow("Create Mask", display_img)

#     # Draw the rectangle permanently on the mask image and update the image display
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         cv2.rectangle(mask, (ix, iy), (x, y), (255, 255, 255), -1)
#         display_img = cv2.addWeighted(img, 0.8, mask, 0.2, 0)
#         cv2.imshow("Create Mask", display_img)


# drawing = False
# ix, iy = -1, -1
# video_path = r"video.mp4"
# video_capture = cv2.VideoCapture(video_path)
# if not video_capture.isOpened():
#     print(f"Failed to open video file: {video_path}")
# else:
#     print("Video file opened successfully.")
#     ret, frame = video_capture.read()
#     if not ret:
#         print("Failed to read the video")
#     else:
#         img = cv2.resize(frame, (800, 600))
#         mask = np.zeros_like(img, dtype=np.uint8)

#         cv2.namedWindow("Create Mask")
#         cv2.setMouseCallback("Create Mask", create_rectangle)

#         display_img = cv2.addWeighted(img, 0.8, mask, 0.2, 0)

#         # Press the ESC key to complete the drawing of the mask
#         while True:
#             cv2.imshow("Create Mask", display_img)
#             k = cv2.waitKey(1) & 0xFF
#             if k == 27:
#                 break

#         # Terminate OpenCV resources when everything is done
#         cv2.imwrite("mask.png", mask)

#         # Save the final mask as a PNG image
#         cv2.destroyAllWindows()
#         video_capture.release()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Previous Attempt 2 - Use of Gaussian blur, edge detection, morphological operations and contours to create the mask #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # Importing essential libraries
# import cv2
# import numpy as np


# # Function to create the mask using a mixture of Gaussian blur, edge detection, morphological operations and contours
# def create_mask(image_path, output_path):
#     # Load the image
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Edge detection
#     edges = cv2.Canny(blurred, 50, 150)

#     # Morphological operations to close the gaps in edges
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

#     # Find contours
#     contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     mask = np.zeros_like(gray)
#     for contour in contours:
#         # Approximate the contour to a polygon
#         perimeter = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

#         # Check if the polygon can be considered a rectangle
#         if len(approx) == 4:
#             _, _, w, h = cv2.boundingRect(approx)
#             aspect_ratio = w / float(h)
#             if aspect_ratio > 0.8 and aspect_ratio < 1.2:
#                 cv2.drawContours(mask, [approx], -1, 255, thickness=cv2.FILLED)

#     # Save the final mask as a PNG image
#     cv2.imwrite(output_path, mask)

#     # Display the output of the operations
#     cv2.imshow("Video Frame Image", img)
#     cv2.imshow("Edges", edges)
#     cv2.imshow("Closed", closed)
#     cv2.imshow("Mask", mask)
#     cv2.waitKey(0)

#     # Terminate OpenCV resources when everything is done
#     cv2.destroyAllWindows()


# # Function call to create a mask using the video frame image and save the resulting mask as a PNG file
# create_mask("video_frame.png", "output_mask.png")

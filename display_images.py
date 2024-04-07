# Import Necessary Libraries
import numpy as np
import cv2


# Define a List of Colors
colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Lime
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 0, 255)   # Pink
        ]


# Define a Function to Display Everything on Both Images
def DisplayImages(left_image, right_image, feature_points, associated_bounding_boxes):
    
    # Display Bounding Boxes on Left & Right Image
    color_index = 0
    for left_bbox, right_bbox in associated_bounding_boxes:
        left_x1, left_y1, left_x2, left_y2 = left_bbox[0], left_bbox[1], left_bbox[2], left_bbox[3]
        right_x1, right_y1, right_x2, right_y2 = right_bbox[0], right_bbox[1], right_bbox[2], right_bbox[3]
        cv2.rectangle(left_image, (left_x1, left_y1), (left_x2, left_y2), colors[color_index], 1)
        cv2.rectangle(right_image, (right_x1, right_y1), (right_x2, right_y2), colors[color_index], 1)
        color_index += 1
    
    # Display Feature Points on Both Images
    for point in feature_points:
        left_x, left_y, right_x, right_y = int(point[0]), int(point[1]), int(point[2]), int(point[3])
        left_image = cv2.circle(left_image, (left_x, left_y), radius = 2, color = (0, 0, 255), thickness = -1)
        right_image = cv2.circle(right_image, (right_x, right_y), radius = 2, color = (0, 0, 255), thickness = -1)

    # Display the Images
    cv2.imshow('Left_Image', left_image)
    cv2.waitKey(1)
    cv2.imshow('Right_Image', right_image)
    cv2.waitKey(1)
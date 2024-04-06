# Import Necessary Libraries
import numpy as np
import cv2


# Define a Function to Display Everything on Both Images
def DisplayImages(left_image, right_image, left_boxes, right_boxes, feature_points):
    
    # Display Bounding Boxes on Left Image
    for bbox in left_boxes:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(left_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Display Bounding Boxes on Right Image
    for bbox in right_boxes:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(right_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
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
# Import Necessary Libraries
import numpy as np
import cv2


# Define a Function to Perform YOLO on Images
def perform_yolo_on_images(model, left_image, right_image):
            
    # Predict the Objects in Left and Right Images using Model
    detections_left_image = model(left_image, verbose = False)
    detections_right_image = model(right_image, verbose = False)

    # Create Result bboxes
    left_bboxes = []
    right_bboxes = []

    # Visualise Bounding Boxes for all Detected Objects in Left Image
    for obj in detections_left_image:
                    
        # Get the Bounding Box Coordinates and Class of Prediction for that Object
        boxes = obj.boxes
        for box in boxes:

            # Convert Bounding Box Coordinates into Array                    
            bbox = np.array(box.xyxy[0].cpu())

            # Get the Coordinates of Bounding Box of Object and Plot on Frame
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            left_bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(left_image, (x1, y1), (x2, y2), (0, 255, 0), 1)


    # Visualise Bounding Boxes for all Detected Objects in Right Image
    for obj in detections_right_image:
                    
        # Get the Bounding Box Coordinates and Class of Prediction for that Object
        boxes = obj.boxes
        for box in boxes:

            # Convert Bounding Box Coordinates into Array                    
            bbox = np.array(box.xyxy[0].cpu())
                    
            # Get the Coordinates of Bounding Box of Object and Plot on Frame
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            right_bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(right_image, (x1, y1), (x2, y2), (0, 255, 0), 1)


    # Display the Detected Objects on Both Images
    cv2.imshow('YOLO on Left_Image', left_image)
    cv2.waitKey(1)
    cv2.imshow('YOLO on Right_Image', right_image)
    cv2.waitKey(1)
    
    # Return the Bounding Boxes
    return left_bboxes, right_bboxes
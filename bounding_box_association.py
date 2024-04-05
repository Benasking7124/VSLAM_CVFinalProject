# Import Necessary Libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Define a Function to Check if a Feature Point falls inside Bounding Box
def check_point_in_bbox(bbox, point):

    # Get the Coordinates and Points
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    x, y = point[0], point[1]

    # Check if Point falls inside Bounding Box
    if x > x1 and x < x2 and y > y1 and y < y2:
        return True
    else:
        return False
    

# Define a Function to Associate Bounding Boxes between Left and Right images
def BoundingBoxAssociation(left_boxes, right_boxes, feature_points):

    # Initialise Empty Dictionary to store Objects in Left and Right Images
    objects_on_left_image = dict()
    objects_on_right_image = dict()

    # Store the Bounding Boxes for both Frames
    objects_on_left_image['Bounding_Boxes'] = dict()
    objects_on_left_image['Bounding_Boxes']['Coordinates'] = left_boxes
    objects_on_right_image['Bounding_Boxes'] = dict()
    objects_on_right_image['Bounding_Boxes']['Coordinates'] = right_boxes

    # Store the Feature Points for both Frames
    objects_on_left_image['Feature_Points'] = feature_points[:, 0: 2]
    objects_on_right_image['Feature_Points'] = feature_points[:, 2: 4]

    # Initialise a List to store Feature Points for Corresponding Bounding Boxes
    objects_on_left_image['Bounding_Boxes']['Feature_Points'] = []
    objects_on_right_image['Bounding_Boxes']['Feature_Points'] = []
    objects_on_left_image['Bounding_Boxes']['Number_of_Feature_Points'] = []
    objects_on_right_image['Bounding_Boxes']['Number_of_Feature_Points'] = []

    # Check every Bounding Box Coordinates in Left Image
    for bbox in objects_on_left_image['Bounding_Boxes']['Coordinates']:

        # Initialise List of Points
        points = []

        # Check every Feature Point in Left Image
        for point in objects_on_left_image['Feature_Points']:
            
            # Check if that Points falls inside Bounding box
            if check_point_in_bbox(bbox, point):
                points.append(point)
        
        # Store the Feature Points for that Bounding Box
        objects_on_left_image['Bounding_Boxes']['Feature_Points'].append(points)
        objects_on_left_image['Bounding_Boxes']['Number_of_Feature_Points'].append(len(points))
    

    # Check every Bounding Box Coordinates in Right Image
    for bbox in objects_on_right_image['Bounding_Boxes']['Coordinates']:

        # Initialise List of Points
        points = []

        # Check every Feature Point in Right Image
        for point in objects_on_right_image['Feature_Points']:
            
            # Check if that Points falls inside Bounding box
            if check_point_in_bbox(bbox, point):
                points.append(point)
        
        # Store the Feature Points for that Bounding Box
        objects_on_right_image['Bounding_Boxes']['Feature_Points'].append(points)
        objects_on_right_image['Bounding_Boxes']['Number_of_Feature_Points'].append(len(points))
    
    print('L', objects_on_left_image['Bounding_Boxes']['Number_of_Feature_Points'])
    print('R', objects_on_right_image['Bounding_Boxes']['Number_of_Feature_Points'])
    
    return 1
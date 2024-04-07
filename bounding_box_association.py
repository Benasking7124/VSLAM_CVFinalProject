# Import Necessary Libraries
import matplotlib.pyplot as plt
import numpy as np
import collections
import cv2
import time


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
    

# Define a Function to Check if Features are Matching
def is_Matching_Features(feature_points, left_feature, right_feature):
    
    # Create Feature from Left and Right Features
    feature = [left_feature[0], left_feature[1], right_feature[0], right_feature[1]]
    
    # Check if Feature is present in List of Feature Points
    for point in feature_points:
        if collections.Counter(point) == collections.Counter(feature):
            return True
    return False


# Define a Function to Associate Bounding Boxes between Left and Right images
def BoundingBoxAssociation(left_image, right_image, left_boxes, right_boxes, feature_points):

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
    

    # Initialise List to store Associated Bounding Boxes
    associated_bounding_boxes = []

    # For every Bounding box in Left Image
    for left_bbox_ind in range(len(objects_on_left_image['Bounding_Boxes']['Coordinates'])):

        # Get the Feature Points and Initialise Matches Count for Features
        left_features = objects_on_left_image['Bounding_Boxes']['Feature_Points'][left_bbox_ind]
        matches = []

        # For every Bounding box in Right Image
        for right_bbox_ind in range(len(objects_on_right_image['Bounding_Boxes']['Coordinates'])):

            # Get the Feature Points and Initialise Matches Count
            right_features = objects_on_right_image['Bounding_Boxes']['Feature_Points'][right_bbox_ind]
            match = 0

            # For every Feature point in Left image
            for left_feature in left_features:

                # For every Feature point in Right image
                for right_feature in right_features:

                    # Check if the Features are Matching Features
                    if is_Matching_Features(feature_points, left_feature, right_feature):
                        match += 1
        
            # Append the Matches count
            matches.append(match)
        
        # Get the Highest Match count and its Index if count > 0
        max_match = max(matches)
        if max_match > 0:
            max_match_index = matches.index(max_match)

            # Append the Matching Bounding Boxes
            associated_bounding_boxes.append([objects_on_left_image['Bounding_Boxes']['Coordinates'][left_bbox_ind], objects_on_right_image['Bounding_Boxes']['Coordinates'][max_match_index]])
    
    # Return Associated Bounding Boxes
    return associated_bounding_boxes
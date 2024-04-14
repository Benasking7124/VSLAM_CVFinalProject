# Import Necessary Libraries
from FeaturePoint import FeaturePoint
import numpy as np
import cv2


# Define a Function to Extract Feature Points
def FeatureExtraction(left_img, right_img):

    # Create an ORB Object to Extract Keypoints
    orb = cv2.ORB_create() 

    # Find the Keypoints of Left and Right Images
    left_keypoints, left_desc = orb.detectAndCompute(left_img, None) 
    right_keypoints, right_desc = orb.detectAndCompute(right_img, None)

    # Define Parameters for Matching Keypoints
    FLANN_INDEX_LSH = 6

    index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,  
                        key_size = 12,   
                        multi_probe_level = 1)  
    
    search_params = dict(checks = 50)  
    
    # Create a Flann Based Matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find Matches and Sort based On Distances
    matches = flann.match(left_desc, right_desc)
    matches = sorted(matches, key = lambda x: x.distance)

    # BFM
    # bfm = cv2.BFMatcher()
    # matches = bfm.match(left_desc, right_desc)
    # matches = sorted(matches, key = lambda x: x.distance)
    # match_img = cv2.drawMatches(left_img, left_keypoints, 
    #                             right_img, right_keypoints, matches,None)
    
    # cv2.imshow('matches', match_img)
    # cv2.waitKey()


    # Compute Feature Points and Disparity for all Matches
    feature_points = []

    # For all Matches
    for m in matches:

        # Get the Indices of Matches
        left_index = m.queryIdx
        right_index = m.trainIdx

        # Compute Feature Points and Disparity
        if (abs(left_keypoints[left_index].pt[1] - right_keypoints[right_index].pt[1]) > 0.00001):
            continue
        
        # Create Feature Point Class object
        feature_point = FeaturePoint()
        
        # Save the Coordinates of Left Image of Feature point
        feature_point.left_pt = (left_keypoints[left_index].pt[0], left_keypoints[left_index].pt[1])
        feature_point.left_descriptor = left_desc[left_index]
        
        # Save the Coordinates of Right Image of Feature point
        feature_point.right_pt = (right_keypoints[right_index].pt[0], right_keypoints[right_index].pt[1])
        feature_point.right_descriptor = right_desc[right_index]

        feature_point.disparity = left_keypoints[left_index].pt[0] - right_keypoints[right_index].pt[0]
        
        # Append Feature point to a List of Points
        feature_points.append(feature_point)

    # Return Feature Points and Disparity
    return feature_points

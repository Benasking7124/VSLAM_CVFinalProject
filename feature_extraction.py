# Import Necessary Libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Define a Function to Extract Feature Points
def featureExtraction(left_img, right_img):

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

    # Compute Feature Points
    feature_points = []   
    for m in matches:
        left_index = m.queryIdx
        right_index = m.trainIdx
        feature_points.append([left_keypoints[left_index].pt[0], left_keypoints[left_index].pt[1], right_keypoints[right_index].pt[0], right_keypoints[right_index].pt[1], (left_keypoints[left_index].pt[0] - right_keypoints[right_index].pt[0])])
    feature_points = np.array(feature_points)

    # Return Feature Points
    return feature_points
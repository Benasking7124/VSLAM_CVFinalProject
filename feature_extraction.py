# Import Necessary Libraries
import numpy as np
import cv2
from FeaturePoint import FeaturePoint

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

    # Compute Feature Points and Disparity for all Matches
    feature_points = []

    # For all Matches
    for m in matches:

        # Get the Indices of Matches
        left_index = m.queryIdx
        right_index = m.trainIdx

        # Compute Feature Points and Disparity
        if (abs(left_keypoints[left_index].pt[1] - right_keypoints[right_index].pt[1]) > 0.001):
            continue
        
        feature_point = FeaturePoint()
        
        feature_point.left_pt = (left_keypoints[left_index].pt[0], left_keypoints[left_index].pt[1])
        feature_point.left_descriptor = left_desc[left_index]
        
        feature_point.right_pt = (right_keypoints[right_index].pt[0], right_keypoints[right_index].pt[1])
        feature_point.right_descriptor = right_desc[right_index]

        feature_point.disparity = left_keypoints[left_index].pt[0] - right_keypoints[right_index].pt[0]
        feature_points.append(feature_point)

    # Convert into Numpy Array
    # feature_points = np.array(feature_points)

    # Return Feature Points and Disparity
    return feature_points

# Old version
# def FeatureExtraction(left_img, right_img):

#     # Create an ORB Object to Extract Keypoints
#     orb = cv2.ORB_create() 

#     # Find the Keypoints of Left and Right Images
#     left_keypoints, left_desc = orb.detectAndCompute(left_img, None) 
#     right_keypoints, right_desc = orb.detectAndCompute(right_img, None)

#     # Define Parameters for Matching Keypoints
#     FLANN_INDEX_LSH = 6

#     index_params = dict(algorithm = FLANN_INDEX_LSH,
#                         table_number = 6,  
#                         key_size = 12,   
#                         multi_probe_level = 1)  
    
#     search_params = dict(checks = 50)  
    
#     # Create a Flann Based Matcher
#     flann = cv2.FlannBasedMatcher(index_params, search_params)

#     # Find Matches and Sort based On Distances
#     matches = flann.match(left_desc, right_desc)
#     matches = sorted(matches, key = lambda x: x.distance)

#     # Compute Feature Points and Disparity for all Matches
#     feature_points = []   
#     disparity = []

#     # For all Matches
#     for m in matches:

#         # Get the Indices of Matches
#         left_index = m.queryIdx
#         right_index = m.trainIdx

#         # Compute Feature Points and Disparity
#         feature_points.append([left_keypoints[left_index].pt[0], left_keypoints[left_index].pt[1], right_keypoints[right_index].pt[0], right_keypoints[right_index].pt[1]])
#         disparity.append([left_keypoints[left_index].pt[0] - right_keypoints[right_index].pt[0]])

#     # Convert into Numpy Array
#     feature_points = np.array(feature_points)
#     disparity = np.array(disparity)

#     # Return Feature Points and Disparity
#     return feature_points, disparity
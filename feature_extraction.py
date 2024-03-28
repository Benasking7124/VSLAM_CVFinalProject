import cv2
import numpy as np 
import matplotlib.pyplot as plt

def featureExtraction(left_img, right_img):

    # Extract keypoints
    orb = cv2.ORB_create() 

    left_keypoints, left_desc = orb.detectAndCompute(left_img,None) 
    right_keypoints, right_desc = orb.detectAndCompute(right_img,None)

    # Matching keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.match(left_desc, right_desc)

    feature_points = []   
    for m in matches:
        left_index = m.queryIdx
        right_index = m.trainIdx

        # featurePoint = [left_kp_x, left_kp_y, right_kp_x, right_kp_y, disparity]
        feature_points.append([left_keypoints[left_index].pt[0], left_keypoints[left_index].pt[1], right_keypoints[right_index].pt[0], right_keypoints[right_index].pt[1], (left_keypoints[left_index].pt[0] - right_keypoints[right_index].pt[0])])
    
    feature_points = np.array(feature_points)

    return feature_points
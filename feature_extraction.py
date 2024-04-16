# Import Necessary Libraries
from FeaturePoint import FeaturePoint
import numpy as np
import cv2

import matplotlib.pyplot as plt


# Define a Function to Extract Feature Points
def FeatureExtraction(left_img, right_img):

    # Create an ORB Object to Extract Keypoints
    orb = cv2.ORB_create(edgeThreshold=51)
    # sift = cv2.SIFT_create()

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
    # bf = cv2.BFMatcher()

    # Find Matches and Sort based On Distances
    matches = flann.knnMatch(left_desc, right_desc,k=2)
    # matches = bf.knnMatch(left_desc, right_desc,k=2)

    keep_matches = []
    for (m, n) in matches:
        if (m.distance / n.distance) < 0.75:
            keep_matches.append(m)

    # match_img_inverse = cv2.drawMatches(left_img, left_keypoints, 
    #                             right_img, right_keypoints, keep_matches,None)
    # cv2.imshow('keep', match_img_inverse)
    
    matches = flann.knnMatch(right_desc, left_desc,k=2)

    keep_matches_inverse = []
    for (m, n) in matches:
        if (m.distance / n.distance) < 0.75:
            keep_matches_inverse.append(m)
    
    # match_img_inverse = cv2.drawMatches(right_img, right_keypoints, left_img, left_keypoints, 
    #                             keep_matches_inverse[:20],None)
    # cv2.imshow('inverse', match_img_inverse)
    
    # cv2.waitKey()

    good_matches = []
    for m in keep_matches:
        for n in keep_matches_inverse:
            if (m.queryIdx == n.trainIdx) and (m.trainIdx == n.queryIdx):
                good_matches.append(m)

    print('Number of matches', len(good_matches))
    
    # Compute Feature Points and Disparity for all Matches

    if len(good_matches)>10:
        src_pts = np.float32([left_keypoints[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([right_keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3)
        matchesMask = mask.ravel().tolist()

        print('Number of inliers', matchesMask.count(1))


    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(left_img,left_keypoints,right_img,right_keypoints,good_matches,None,**draw_params)


    # plt.imshow(img3, 'gray')
    # plt.show()


    outliers = []

    for one_case in matchesMask:
        if one_case == 0:
            outliers.append(1)
        else:
            outliers.append(0)

    print('Number of outliers', outliers.count(1))

    draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = outliers, # draw only inliers
                flags = 2)

    img3 = cv2.drawMatches(left_img,left_keypoints,right_img,right_keypoints,good_matches,None,**draw_params)

    # plt.imshow(img3, 'gray')
    # plt.show()


    filtered_values = [value for m, value in zip(matchesMask, good_matches) if m == 1]


    feature_points = []

    # For all Matches
    # for m in good_matches:
    for m in filtered_values:

        # Get the Indices of Matches
        left_index = m.queryIdx
        right_index = m.trainIdx

        # Compute Feature Points and Disparity
        if (abs(left_keypoints[left_index].pt[1] - right_keypoints[right_index].pt[1]) > 0.0000001):
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

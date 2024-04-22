import cv2
import numpy as np
from itertools import compress

def feature_matching(keypoints1, descriptors1, keypoints2, descrptors2):
    """
    Match feature points from two images
    
    @param {tuple} keypoints1 - a series of keypoint(type: cv2.KeyPoint) from the first image
    @param {numpy.ndarray} descriptors1 - a 2d array of descriptors from the first image
    @param {tuple} keypoints2 - a series of keypoint(type: cv2.KeyPoint) from the second image
    @param {numpy.ndarray} descrptors2 - a 2d array of descriptors from the second image

    @return {cv2.DMatch} a list of DMatch objects
    """
    
    # Define Parameters for Flann Matching
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)

    search_params = dict(checks = 50)

    # Create a Flann Based Matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Matching from 1 to 2
    matches_1_2 = flann.knnMatch(descriptors1, descrptors2, k = 2)
    
    # Matching from 1 to 2
    matches_2_1 = flann.knnMatch(descrptors2, descriptors1, k = 2)




    # ------ Some of matches desc1->desc2 or desc2->desc1 are tuple (length 0 or length 1) which means not pair... --
    # ------ Code block below detect such cases and save in the list for deleting -----------------------------------
    matches_1_2 = list(matches_1_2)
    indices_for_delete_1_2 = []
    for i, one_match in enumerate(matches_1_2):
        if len(one_match)==0 or len(one_match)==1:
            indices_for_delete_1_2.append(i)
        
    matches_2_1 = list(matches_2_1)
    indices_for_delete_2_1 = []

    for i, one_match in enumerate(matches_2_1):
         if len(one_match)==0 or len(one_match)==1:
            indices_for_delete_2_1.append(i)

    # Deleting length 0 or length 1 tuples(matches)
    all_indicies_deletes = indices_for_delete_1_2+indices_for_delete_2_1
    all_indicies_deletes = list(set(all_indicies_deletes))

    for index in sorted(all_indicies_deletes, reverse=True):
        del matches_1_2[index]
        del matches_2_1[index]

    matches_1_2 = tuple(matches_1_2)
    matches_2_1 = tuple(matches_2_1)
    # -------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------




    # Ratio Test 1->2
    keep_matches_1 = []
    for (m, n) in matches_1_2:
        if (m.distance / n.distance) < 0.7:
            keep_matches_1.append(m)


    # Ratio Test 2->1
    keep_matches_2 = []
    for (m, n) in matches_2_1:
        if (m.distance / n.distance) < 0.7:
            keep_matches_2.append(m)

    # Bi-directional Check
    bi_direction_matches = []
    for m in keep_matches_1:
        for n in keep_matches_2:
            if (m.queryIdx == n.trainIdx) and (m.trainIdx == n.queryIdx):
                bi_direction_matches.append(m)

    # Do RANSAC to filter out outliers
    if len(bi_direction_matches)>10:
        src_pts = np.float32([[keypoints1[m.queryIdx][0], keypoints1[m.queryIdx][1]] for m in bi_direction_matches])
        dst_pts = np.float32([[keypoints2[m.trainIdx][0], keypoints2[m.trainIdx][1]] for m in bi_direction_matches])

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)
        matchesMask = mask.ravel().tolist()
        good_matches = list(compress(bi_direction_matches, matchesMask))
    else:
        good_matches = None


    return good_matches
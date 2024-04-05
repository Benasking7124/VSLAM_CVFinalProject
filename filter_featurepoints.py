import cv2
import numpy as np

def filter_featurepointsbydepth(featurePoints,left_image, right_image):

    # Define Parameters
    focal_length = 718.856
    baseline = 0.54
    depth_threshold = 10

    # Compute Disparity and Depth Map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity_map = stereo.compute(left_image, right_image)

    disparity_map[disparity_map == 0] = 0.1
    depth_map = (focal_length * baseline) / disparity_map

    # Filter feature points based on depth threshold
    filtered_feature_points = []

    for point in featurePoints:
        x, y = int(point.pt[0]), int(point.pt[1])
        depth = depth_map[y, x]
        if depth < depth_threshold:
            filtered_feature_points.append(point)

    return np.array(filtered_feature_points)

import cv2
import numpy as np

def filter_featurepointsbydepth(featurePoints, left_image, right_image):

    # Define Parameters
    focal_length = 718.856
    baseline = 0.54
    depth_threshold = 10000

    # Compute Disparity and Depth Map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity_map = stereo.compute(cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY))
    print(disparity_map)

    disparity_map[disparity_map == 0] = 0.1
    depth_map = (focal_length * baseline) / disparity_map

    # Filter feature points based on depth threshold
    filtered_feature_points = []

    for point in featurePoints:
        left_x, left_y, right_x, right_y = int(point[0]), int(point[1]), int(point[2]), int(point[3])
        depth = depth_map[left_y, left_x]    
        if depth < depth_threshold:
            filtered_feature_points.append(point)

    return np.array(filtered_feature_points)

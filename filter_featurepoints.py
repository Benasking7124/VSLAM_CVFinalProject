import cv2
import numpy as np

def filter_featurepointsbydepth(featurePoints, depth_map):
    depth_threshold = 1000
    filteredFeaturePoints = []
    for point in featurePoints:
        left_x, left_y, right_x, right_y = int(point[0]), int(point[1]), int(point[2]), int(point[3])
        depth = depth_map[left_y, left_x]
        if depth < depth_threshold:
            filteredFeaturePoints.append(point)
    return np.array(filteredFeaturePoints)

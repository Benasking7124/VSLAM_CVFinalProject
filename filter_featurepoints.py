import numpy as np

def get_filtered_feature_points(featurePoints, depth_map):
    depth_threshold = 1000
    filteredFeaturePoints = []
    for point in featurePoints:
        x, y = int(point.pt[0]), int(point.pt[1])
        depth = depth_map[y, x]
        if depth < depth_threshold:
            filteredFeaturePoints.append(point)
    return np.array(filteredFeaturePoints)

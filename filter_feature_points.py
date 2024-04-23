# Import Necessary Libraries
import numpy as np
from sklearn.cluster import KMeans


# Define a Function to Filter Feature Points by Depth
def FilterFeaturePoints(featurePoints, depth_map, num_clusters=2):
    
    feature_depths = []
    feature_coords = []

    # Collect depth information for each feature point
    for point in featurePoints:
        left_x, left_y, right_x, right_y = int(point.left_pt[0]), int(point.left_pt[1]), int(point.right_pt[0]), int(point.right_pt[1])
        depth_pos = [int((left_x + right_x) / 2), int((left_y + right_y) / 2)]
        point.depth = depth_map[depth_pos[1], depth_pos[0]]
        feature_depths.append(point.depth)
        feature_coords.append([left_x, left_y, right_x, right_y])

    # Convert depth list to numpy array for clustering
    feature_depths = np.array(feature_depths).reshape(-1, 1)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feature_depths)
    labels = kmeans.labels_

    # Initialise lists to store static and dynamic feature points
    static_feature_points = []
    dynamic_feature_points = []

    # Classify points based on clusters
    for idx, label in enumerate(labels):
        if label == 0:
            static_feature_points.append(feature_coords[idx] + [feature_depths[idx, 0]])
        else:
            dynamic_feature_points.append(feature_coords[idx] + [feature_depths[idx, 0]])


    static_feature_points = np.array(static_feature_points)
    dynamic_feature_points = np.array(dynamic_feature_points)
    
    return static_feature_points, dynamic_feature_points

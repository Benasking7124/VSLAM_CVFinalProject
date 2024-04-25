# Import Necessary Libraries
import numpy as np
from sklearn.cluster import KMeans
from feature_points import FeaturePoints


# Define a Function to Filter Feature Points by Depth
def FilterFeaturePoints(feature_points, num_clusters = 2):
    
    # Initialise List to Store Feature Depths and Coordinates
    feature_depths = []
    feature_coords = []

    # Store Depth and Coordinates for every Feature Point
    for ind in range(feature_points.num_fp):

        # Get the Left and Right Coordinates
        left_x, left_y, right_x, right_y = int(feature_points.left_pts[ind][0]), int(feature_points.left_pts[ind][1]), int(feature_points.right_pts[ind][0]), int(feature_points.right_pts[ind][1])
        feature_depths.append(feature_points.depth[ind])
        feature_coords.append([left_x, left_y, right_x, right_y])

    # Convert depth list to numpy array for clustering
    feature_depths = np.array(feature_depths).reshape(-1, 1)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters = num_clusters, random_state = 42).fit(feature_depths)
    labels = kmeans.labels_

    # Initialise Static Feature points
    static_feature_points = FeaturePoints()
    static_feature_points.left_pts = np.empty([0, 2])
    static_feature_points.left_descriptors = np.empty([0, 32], dtype = np.uint8)
    static_feature_points.right_pts = np.empty([0, 2])
    static_feature_points.right_descriptors = np.empty([0, 32], dtype = np.uint8)
    static_feature_points.disparity = np.empty([0, 1])
    static_feature_points.depth = np.empty([0, 1])
    static_feature_points.pt3ds = np.empty([0, 3])

    # Initialise Dynamic Feature points
    dynamic_feature_points = FeaturePoints()
    dynamic_feature_points.left_pts = np.empty([0, 2])
    dynamic_feature_points.left_descriptors = np.empty([0, 32], dtype = np.uint8)
    dynamic_feature_points.right_pts = np.empty([0, 2])
    dynamic_feature_points.right_descriptors = np.empty([0, 32], dtype = np.uint8)
    dynamic_feature_points.disparity = np.empty([0, 1])
    dynamic_feature_points.depth = np.empty([0, 1])
    dynamic_feature_points.pt3ds = np.empty([0, 3])

    # Classify points based on clusters
    for idx, label in enumerate(labels):
        if label == 1:
            static_feature_points.left_pts = np.vstack([static_feature_points.left_pts, feature_points.left_pts[idx]])
            static_feature_points.left_descriptors = np.vstack([static_feature_points.left_descriptors, feature_points.left_descriptors[idx]])
            static_feature_points.right_pts = np.vstack([static_feature_points.right_pts, feature_points.right_pts[idx]])
            static_feature_points.right_descriptors = np.vstack([static_feature_points.right_descriptors, feature_points.right_descriptors[idx]])
            static_feature_points.disparity = np.vstack([static_feature_points.disparity, feature_points.disparity[idx]])
            static_feature_points.depth = np.vstack([static_feature_points.depth, feature_points.depth[idx]])
            static_feature_points.pt3ds = np.vstack([static_feature_points.pt3ds, feature_points.pt3ds[idx]])
        else:
            dynamic_feature_points.left_pts = np.vstack([dynamic_feature_points.left_pts, feature_points.left_pts[idx]])
            dynamic_feature_points.left_descriptors = np.vstack([dynamic_feature_points.left_descriptors, feature_points.left_descriptors[idx]])
            dynamic_feature_points.right_pts = np.vstack([dynamic_feature_points.right_pts, feature_points.right_pts[idx]])
            dynamic_feature_points.right_descriptors = np.vstack([dynamic_feature_points.right_descriptors, feature_points.right_descriptors[idx]])
            dynamic_feature_points.disparity = np.vstack([dynamic_feature_points.disparity, feature_points.disparity[idx]])
            dynamic_feature_points.depth = np.vstack([dynamic_feature_points.depth, feature_points.depth[idx]])
            dynamic_feature_points.pt3ds = np.vstack([dynamic_feature_points.pt3ds, feature_points.pt3ds[idx]])
    
    # Set the Number of Feature Points
    static_feature_points.num_fp = static_feature_points.left_pts.shape[0]
    dynamic_feature_points.num_fp = dynamic_feature_points.left_pts.shape[0]
    
    # Return the Static and Dynamic Feature Points
    return static_feature_points, dynamic_feature_points

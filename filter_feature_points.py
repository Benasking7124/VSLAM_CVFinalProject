# Import Necessary Libraries
import numpy as np
from sklearn.cluster import KMeans
from feature_points import FeaturePoints

def check_point_in_bbox(bbox, point):
    # Get the Coordinates and Points
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    x, y = point[0], point[1]

    # Check if Point falls inside Bounding Box
    if x > x1 and x < x2 and y > y1 and y < y2:
        return True
    else:
        return False



# Define a Function to Filter Feature Points by Depth
def FilterFeaturePoints(left_boxes, right_boxes, feature_points, num_clusters = 2):
    
    # Initialise Static Feature points
    static_feature_points = FeaturePoints()
    static_feature_points.left_pts = np.empty([0, 2])
    static_feature_points.left_descriptors = np.empty([0, 32], dtype = np.uint8)
    static_feature_points.right_pts = np.empty([0, 2])
    static_feature_points.right_descriptors = np.empty([0, 32], dtype = np.uint8)
    static_feature_points.disparity = np.empty([0, 1])
    static_feature_points.depth = np.empty([0, 1])
    static_feature_points.pt3ds = np.empty([0, 3])

    # Initialise Points Inside Bounding Box
    points_inside_bounding_box = FeaturePoints()
    points_inside_bounding_box.left_pts = np.empty([0, 2])
    points_inside_bounding_box.left_descriptors = np.empty([0, 32], dtype = np.uint8)
    points_inside_bounding_box.right_pts = np.empty([0, 2])
    points_inside_bounding_box.right_descriptors = np.empty([0, 32], dtype = np.uint8)
    points_inside_bounding_box.disparity = np.empty([0, 1])
    points_inside_bounding_box.depth = np.empty([0, 1])
    points_inside_bounding_box.pt3ds = np.empty([0, 3])

    for idx in range(feature_points.num_fp):
        left_point = feature_points.left_pts[idx]
        right_point = feature_points.right_pts[idx]
        for left_box in left_boxes:
            # Classify points based on clusters
            if check_point_in_bbox(left_box, left_point):
                points_inside_bounding_box.left_pts = np.vstack([points_inside_bounding_box.left_pts, feature_points.left_pts[idx]])
                points_inside_bounding_box.left_descriptors = np.vstack([points_inside_bounding_box.left_descriptors, feature_points.left_descriptors[idx]])
                points_inside_bounding_box.disparity = np.vstack([points_inside_bounding_box.disparity, feature_points.disparity[idx]])
                points_inside_bounding_box.depth = np.vstack([points_inside_bounding_box.depth, feature_points.depth[idx]])
                points_inside_bounding_box.pt3ds = np.vstack([points_inside_bounding_box.pt3ds, feature_points.pt3ds[idx]])
            else:
                static_feature_points.left_pts = np.vstack([static_feature_points.left_pts, feature_points.left_pts[idx]])
                static_feature_points.left_descriptors = np.vstack([static_feature_points.left_descriptors, feature_points.left_descriptors[idx]])
                static_feature_points.disparity = np.vstack([static_feature_points.disparity, feature_points.disparity[idx]])
                static_feature_points.depth = np.vstack([static_feature_points.depth, feature_points.depth[idx]])
                static_feature_points.pt3ds = np.vstack([static_feature_points.pt3ds, feature_points.pt3ds[idx]])

        for right_box in right_boxes:
            # Classify points based on clusters
            if check_point_in_bbox(right_box, right_point):
                points_inside_bounding_box.right_pts = np.vstack([points_inside_bounding_box.right_pts, feature_points.right_pts[idx]])
                points_inside_bounding_box.right_descriptors = np.vstack([points_inside_bounding_box.right_descriptors, feature_points.right_descriptors[idx]])
                points_inside_bounding_box.disparity = np.vstack([points_inside_bounding_box.disparity, feature_points.disparity[idx]])
                points_inside_bounding_box.depth = np.vstack([points_inside_bounding_box.depth, feature_points.depth[idx]])
                points_inside_bounding_box.pt3ds = np.vstack([points_inside_bounding_box.pt3ds, feature_points.pt3ds[idx]])
            else:
                static_feature_points.right_pts = np.vstack([static_feature_points.right_pts, feature_points.right_pts[idx]])
                static_feature_points.right_descriptors = np.vstack([static_feature_points.right_descriptors, feature_points.right_descriptors[idx]])
                static_feature_points.disparity = np.vstack([static_feature_points.disparity, feature_points.disparity[idx]])
                static_feature_points.depth = np.vstack([static_feature_points.depth, feature_points.depth[idx]])
                static_feature_points.pt3ds = np.vstack([static_feature_points.pt3ds, feature_points.pt3ds[idx]])
    
    points_inside_bounding_box.num_fp = np.minimum(points_inside_bounding_box.left_pts.shape[0], points_inside_bounding_box.right_pts.shape[0])
    static_feature_points.num_fp = np.minimum(static_feature_points.left_pts.shape[0], static_feature_points.right_pts.shape[0])

    # Initialise List to Store Feature Depths and Coordinates
    feature_depths = []
    feature_coords = []

    dynamic_feature_points = FeaturePoints()
    dynamic_feature_points.left_pts = np.empty([0, 2])
    dynamic_feature_points.left_descriptors = np.empty([0, 32], dtype = np.uint8)
    dynamic_feature_points.right_pts = np.empty([0, 2])
    dynamic_feature_points.right_descriptors = np.empty([0, 32], dtype = np.uint8)
    dynamic_feature_points.disparity = np.empty([0, 1])
    dynamic_feature_points.depth = np.empty([0, 1])
    dynamic_feature_points.pt3ds = np.empty([0, 3])
    # Store Depth and Coordinates for every Feature Point
    for ind in range(points_inside_bounding_box.num_fp):

        # Get the Left and Right Coordinates
        left_x, left_y, right_x, right_y = int(points_inside_bounding_box.left_pts[ind][0]), int(points_inside_bounding_box.left_pts[ind][1]), int(points_inside_bounding_box.right_pts[ind][0]), int(points_inside_bounding_box.right_pts[ind][1])
        feature_depths.append(points_inside_bounding_box.depth[ind])
        feature_coords.append([left_x, left_y, right_x, right_y])

    # Convert depth list to numpy array for clustering
    feature_depths = np.array(feature_depths).reshape(-1, 1)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters = num_clusters, random_state = 42).fit(feature_depths)
    labels = kmeans.labels_

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
    return static_feature_points, points_inside_bounding_box

# Import Necessary Modules
from feature_points import FeaturePoints

# Import Necessary Libraries
import numpy as np
from sklearn.cluster import KMeans


# Define a Function to Check if a Point lies in a Bounding Box
def check_point_in_bbox(bbox, point):

    # Get the Coordinates and Points
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    x, y = point[0], point[1]

    # Check if Point falls inside Bounding Box
    if x > x1 and x < x2 and y > y1 and y < y2:
        return True
    else:
        return False


# Define a Function to Filter Feature Points by Depth using K-Means
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

    # Initialise Dynamic
    dynamic_feature_points = FeaturePoints()
    dynamic_feature_points.left_pts = np.empty([0, 2])
    dynamic_feature_points.left_descriptors = np.empty([0, 32], dtype = np.uint8)
    dynamic_feature_points.right_pts = np.empty([0, 2])
    dynamic_feature_points.right_descriptors = np.empty([0, 32], dtype = np.uint8)
    dynamic_feature_points.disparity = np.empty([0, 1])
    dynamic_feature_points.depth = np.empty([0, 1])
    dynamic_feature_points.pt3ds = np.empty([0, 3])

    # For every Feature Point
    for idx in range(feature_points.num_fp):

        # Get the Left and Right Point
        left_point = feature_points.left_pts[idx]
        right_point = feature_points.right_pts[idx]
        left_found = False
        right_found = False

        # For every Bounding Box in Left Image
        for left_box in left_boxes:

            # If Point lies inside Bounding Box, Append the Points into Bbox List
            if check_point_in_bbox(left_box, left_point):
                dynamic_feature_points.left_pts = np.vstack([dynamic_feature_points.left_pts, feature_points.left_pts[idx]])
                dynamic_feature_points.left_descriptors = np.vstack([dynamic_feature_points.left_descriptors, feature_points.left_descriptors[idx]])
                dynamic_feature_points.disparity = np.vstack([dynamic_feature_points.disparity, feature_points.disparity[idx]])
                dynamic_feature_points.depth = np.vstack([dynamic_feature_points.depth, feature_points.depth[idx]])
                dynamic_feature_points.pt3ds = np.vstack([dynamic_feature_points.pt3ds, feature_points.pt3ds[idx]])
                
                # Set flag and Break Loop
                left_found = True
                break
        
        # If Point not lies in any Bounding Box, Append the Points into Static
        if not left_found:
            static_feature_points.left_pts = np.vstack([static_feature_points.left_pts, feature_points.left_pts[idx]])
            static_feature_points.left_descriptors = np.vstack([static_feature_points.left_descriptors, feature_points.left_descriptors[idx]])
            static_feature_points.disparity = np.vstack([static_feature_points.disparity, feature_points.disparity[idx]])
            static_feature_points.depth = np.vstack([static_feature_points.depth, feature_points.depth[idx]])
            static_feature_points.pt3ds = np.vstack([static_feature_points.pt3ds, feature_points.pt3ds[idx]])

        # For every Bounding Box in Right Image
        for right_box in right_boxes:

            # If Point lies inside Bounding Box, Append the Points into Bbox List
            if check_point_in_bbox(right_box, right_point):
                dynamic_feature_points.right_pts = np.vstack([dynamic_feature_points.right_pts, feature_points.right_pts[idx]])
                dynamic_feature_points.right_descriptors = np.vstack([dynamic_feature_points.right_descriptors, feature_points.right_descriptors[idx]])
                dynamic_feature_points.disparity = np.vstack([dynamic_feature_points.disparity, feature_points.disparity[idx]])
                dynamic_feature_points.depth = np.vstack([dynamic_feature_points.depth, feature_points.depth[idx]])
                dynamic_feature_points.pt3ds = np.vstack([dynamic_feature_points.pt3ds, feature_points.pt3ds[idx]])
                
                # Set flag and Break Loop
                right_found = True
                break
        
        # If Point not lies in any Bounding Box, Append the Points into Static
        if not right_found:
            static_feature_points.right_pts = np.vstack([static_feature_points.right_pts, feature_points.right_pts[idx]])
            static_feature_points.right_descriptors = np.vstack([static_feature_points.right_descriptors, feature_points.right_descriptors[idx]])
            static_feature_points.disparity = np.vstack([static_feature_points.disparity, feature_points.disparity[idx]])
            static_feature_points.depth = np.vstack([static_feature_points.depth, feature_points.depth[idx]])
            static_feature_points.pt3ds = np.vstack([static_feature_points.pt3ds, feature_points.pt3ds[idx]])
    
    # Set the Size of Feature Point Classes
    dynamic_feature_points.num_fp = np.minimum(dynamic_feature_points.left_pts.shape[0], dynamic_feature_points.right_pts.shape[0])
    static_feature_points.num_fp = np.minimum(static_feature_points.left_pts.shape[0], static_feature_points.right_pts.shape[0])

    # Return the Static and Dynamic Feature Points
    return static_feature_points, dynamic_feature_points

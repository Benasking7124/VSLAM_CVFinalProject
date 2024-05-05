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


# Define a Function to Filter Feature Points
def FilterFeaturePoints(left_boxes, right_boxes, feature_points):
        
    # Initialise Static Feature points
    static_feature_points = FeaturePoints()
    static_feature_points.left_pts = np.empty([0, 2])
    static_feature_points.left_descriptors = np.empty([0, 32], dtype = np.uint8)
    static_feature_points.right_pts = np.empty([0, 2])
    static_feature_points.right_descriptors = np.empty([0, 32], dtype = np.uint8)
    static_feature_points.disparity = np.empty([0, 1])
    static_feature_points.depth = np.empty([0, 1])
    static_feature_points.pt3ds = np.empty([0, 3])
    # static_feature_points.bbox_id = np.empty([0, 1])

    # Initialise Points Inside Bounding Box
    points_inside_bbox = FeaturePoints()
    points_inside_bbox.left_pts = np.empty([0, 2])
    points_inside_bbox.left_descriptors = np.empty([0, 32], dtype = np.uint8)
    points_inside_bbox.right_pts = np.empty([0, 2])
    points_inside_bbox.right_descriptors = np.empty([0, 32], dtype = np.uint8)
    points_inside_bbox.disparity = np.empty([0, 1])
    points_inside_bbox.depth = np.empty([0, 1])
    points_inside_bbox.pt3ds = np.empty([0, 3])
    # points_inside_bbox.bbox_id = np.empty([0, 1])


    # For every Feature Point
    for idx in range(feature_points.num_fp):

        # Get the Left and Right Point
        left_point = feature_points.left_pts[idx]
        right_point = feature_points.right_pts[idx]
        is_in_bb = False

        # Because sometimes Yolo detects different numbers of bounding boxes from the left and right images, 
        # the number of points outside(or inside) boxes from the left image is not equal to 
        # the number of points outside(or inside) the boxes in the right image.

        # This problem needs to be corrected to calculate correct depth.
        # Therefore, the following codes do: checking whether at least one of the points between the left point and the right point is inside of the box, 
        # and if this is the case, both points will be assigned as potential dynamics.

        for one_left_box in left_boxes:
            
            if check_point_in_bbox(one_left_box, left_point):

                is_in_bb = True
        
        for one_right_box in right_boxes:

            if check_point_in_bbox(one_right_box, right_point):

                is_in_bb = True

        if is_in_bb:
                
                points_inside_bbox.left_pts = np.vstack([points_inside_bbox.left_pts, feature_points.left_pts[idx]])
                points_inside_bbox.left_descriptors = np.vstack([points_inside_bbox.left_descriptors, feature_points.left_descriptors[idx]])

                points_inside_bbox.right_pts = np.vstack([points_inside_bbox.right_pts, feature_points.right_pts[idx]])
                points_inside_bbox.right_descriptors = np.vstack([points_inside_bbox.right_descriptors, feature_points.right_descriptors[idx]])

                points_inside_bbox.disparity = np.vstack([points_inside_bbox.disparity, feature_points.disparity[idx]])
                points_inside_bbox.depth = np.vstack([points_inside_bbox.depth, feature_points.depth[idx]])
                points_inside_bbox.pt3ds = np.vstack([points_inside_bbox.pt3ds, feature_points.pt3ds[idx]])

        else:

                static_feature_points.left_pts = np.vstack([static_feature_points.left_pts, feature_points.left_pts[idx]])
                static_feature_points.left_descriptors = np.vstack([static_feature_points.left_descriptors, feature_points.left_descriptors[idx]])

                static_feature_points.right_pts = np.vstack([static_feature_points.right_pts, feature_points.right_pts[idx]])
                static_feature_points.right_descriptors = np.vstack([static_feature_points.right_descriptors, feature_points.right_descriptors[idx]])

                static_feature_points.disparity = np.vstack([static_feature_points.disparity, feature_points.disparity[idx]])
                static_feature_points.depth = np.vstack([static_feature_points.depth, feature_points.depth[idx]])
                static_feature_points.pt3ds = np.vstack([static_feature_points.pt3ds, feature_points.pt3ds[idx]])

    # Set the Size of Feature Point Classes
    points_inside_bbox.num_fp = np.minimum(points_inside_bbox.left_pts.shape[0], points_inside_bbox.right_pts.shape[0])
    static_feature_points.num_fp = np.minimum(static_feature_points.left_pts.shape[0], static_feature_points.right_pts.shape[0])

    return static_feature_points, points_inside_bbox
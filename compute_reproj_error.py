# Import Necessary Libraries
from feature_points import FeaturePoints
from feature_matching import FeatureMatching
import numpy as np
import cv2


# Define a Function to Perform Inverse Projection from 2D Image Coordinates to 3D Camera Coordinates
def inverse_project_points(p_2d, depth, camera_matrix):
    
    # Get the Inverse of Camera Projection Matrix
    inv_camera_matrix = np.linalg.inv(camera_matrix[:, :3])

    # Compute 2D Homogenous Coordinates
    homogeneous_2d = np.array([p_2d[0], p_2d[1], 1]).T
    return (inv_camera_matrix @ (homogeneous_2d * depth)).T
                                 

# Define a Function to Extract Feature Points
def ComputeReprojError(feature_points, camera_param):
    
    """
    Calculates the reprojection error for a feature point.

    Args:
        P_l_I_i (numpy.ndarray): 2D feature point in image i.
        T_c_w_j (numpy.ndarray): Transform from world to camera frame j.
        T_c_w_i (numpy.ndarray): Transform from world to camera frame i.
        P_n_w (numpy.ndarray): 3D point in world coordinates.
        z_n_c_i (float): Depth of the 3D point in camera frame i.
        K (numpy.ndarray): Camera intrinsic matrix.

    Returns:
        numpy.ndarray: Reprojection error (2D vector).
    """
    image_coordinates = np.empty([0, 3])
    for ind in range(feature_points.num_fp):
        image_coordinates = np.vstack([image_coordinates, [inverse_project_points(feature_points.left_pts[ind], feature_points.depth[ind], camera_param['left_projection'])]])
    print(image_coordinates)

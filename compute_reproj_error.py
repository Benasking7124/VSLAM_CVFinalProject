# Import Necessary Libraries
import numpy as np
import cv2


# Define a Function to Perform Inverse Projection from 2D Image Coordinates to 3D Camera Coordinates
def inverse_project_points(p_2d, depth, camera_matrix):
    
    # Get the Inverse of Camera Projection Matrix
    inv_camera_matrix = np.linalg.inv(camera_matrix[:, :3])

    # Compute 2D Homogenous Coordinates
    homogeneous_2d = np.array([p_2d[0], p_2d[1], 1]).T

    # Compute and Return 3D Camera Coordinates
    camera_coordinates = (inv_camera_matrix @ (homogeneous_2d * depth)).T
    return camera_coordinates
                                 

# Define a Function to Compute Reprojection Error
def ComputeReprojError(feature_points, camera_param):
    
    """
    Calculates the reprojection error for a feature point.

    Args:
        feature_points (numpy.ndarray): 2D feature point in current image.
        T_c_w_j (numpy.ndarray): Transform from world to next camera frame
        T_c_w_i (numpy.ndarray): Transform from world to current camera frame
        P_n_w (numpy.ndarray): 3D point in world coordinates.
        z_n_c_i (float): Depth of the 3D point in camera frame i.
        camera_param (numpy.ndarray): Camera intrinsic matrix.

    Returns:
        numpy.ndarray: Reprojection error (2D vector).
    """

    # Initialise Vectors
    image_coordinates = np.empty([0, 3])
    T_c_w_i = np.identity(3)

    # For every Feature Point
    for ind in range(feature_points.num_fp):

        # Compute 3D Camera Coordinates from 2D Image Coordinates
        current_image_coordinates = np.vstack([image_coordinates, [inverse_project_points(feature_points.left_pts[ind], feature_points.depth[ind], camera_param['left_projection'])]])

        # Transform 3D Camera Coordinates from Current Frame to World Frame
        current_world_coordinates = (T_c_w_i @ current_image_coordinates.T).T

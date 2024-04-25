# Import Necessary Libraries
import numpy as np

# Define a Function to Compute Reprojection Error
def ComputeReprojError(static_feature_points, camera_param, T_previous, T_current):
 
    temp = np.vstack((T_previous.reshape(3, 4), [0, 0, 0, 1])) @ np.hstack((static_feature_points[2:5], [1]))
    temp_2 = np.linalg.inv(np.vstack((T_current.reshape(3, 4), [0, 0, 0, 1])))@temp

    M = camera_param['left_projection']

    projected_homogeneous = M@temp_2

    reprojection_error = np.linalg.norm(static_feature_points[0:2] - projected_homogeneous)

    return reprojection_error

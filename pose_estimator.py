# Import Necessary Libraries
import numpy as np
from scipy.optimize import least_squares

class PoseEstimator:
    def __init__(self, paired_static_features, camera_projection, T_previous) -> None:
        # Stack all the 3D points into one matrix
        self.pt3Ds = paired_static_features[:, 2:5].T
        self.pt3Ds = np.vstack([self.pt3Ds, np.ones([1, self.pt3Ds.shape[1]])])
        
        # Stack all 2d points into one matrix
        self.u_current = paired_static_features[:, 0].flatten()
        self.v_current = paired_static_features[:, 1].flatten()

        # Get the projection matrix
        self.projection = camera_projection

        # Get the previous transfomation matrix
        self.T_previous = T_previous.reshape(3, 4)

        # Set the current transformatnio matrix to identity matrix
        self.T_current = np.eye(4, 4).flatten()

    # Define a Function to Compute Reprojection Error
    def ComputeReprojError(self, T_current):
    
        temp = np.vstack((self.T_previous.reshape(3, 4), [0, 0, 0, 1])) @ self.pt3Ds
        temp_2 = np.linalg.inv(T_current.reshape(4, 4)) @ temp

        projected_homogeneous = self.projection @ temp_2

        u_predict = projected_homogeneous[0] / projected_homogeneous[2]
        v_predict = projected_homogeneous[1] / projected_homogeneous[2]

        reprojection_error = sum(np.sqrt((self.u_current - u_predict) ** 2 + (self.v_current - v_predict) ** 2))

        return reprojection_error
    
    def minimize_error(self):
        self.T_current = least_squares(self.ComputeReprojError, self.T_current).x

        return self.T_current


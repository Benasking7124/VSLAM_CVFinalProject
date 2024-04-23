from FeaturePoints import FeaturePoints
from feature_matching import feature_matching
import numpy as np

def frame_matching(previous_keypoints, current_keypoints):
    """
    Match current feature points with previous featrue points

    @param {FeaturePoints class} previous_keypoints - FeaturePoints object from the previous frame
    @param {FeaturePoints class} current_keypoints - FeaturePoints object from the current frame

    @return {numpy.array} a numpy array of shape (n, 5), consists of paired featurepoints in the format [u_current, v_current, x_previous, y_previous, z_previous]
    """
    
    # Match the Left and Right Keypoints
    left_matches = feature_matching(previous_keypoints.left_pts, previous_keypoints.left_descriptors, current_keypoints.left_pts, current_keypoints.left_descriptors)
    right_matches = feature_matching(previous_keypoints.right_pts, previous_keypoints.right_descriptors, current_keypoints.right_pts, current_keypoints.right_descriptors)

    # Bi-image Check
    good_matches = []
    for i in range(len(left_matches)):
        for j in range(len(right_matches)):
            if (left_matches[i].queryIdx == right_matches[j].queryIdx) and (left_matches[i].trainIdx == right_matches[j].trainIdx):
                good_matches.append(left_matches[i])

    # Return Paired Current 2D Point and Previous 3D Point in a Numpy Array
    paired_feature_points = np.empty([0, 5])
    for m in good_matches:
        current_2d = current_keypoints.left_pts[m.trainIdx]
        previous_3d = previous_keypoints.pt3ds[m.queryIdx]
        
        # [u_current, v_current, X_previous, Y_previous, Z_previous]
        paired_feature_points = np.vstack([paired_feature_points, [current_2d[0], current_2d[1], previous_3d[0], previous_3d[1], previous_3d[2]]])
    
    return paired_feature_points
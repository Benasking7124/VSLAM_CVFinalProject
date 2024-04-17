# Import Necessary Libraries
from FeaturePoint import FeaturePoint
from FeaturePoints import FeaturePoints
from feature_matching import feature_matching
import numpy as np
import cv2

# Define a Function to Extract Feature Points
def feature_extraction(left_img, right_img, camera_param):
    """
    Extract the features in the time frame and reconstruct the feature points

    @param {numpy.ndarray} left_img - a image read by cv2.imread
    @param {numpy.ndarray} right_img - a image read by cv2.imread
    @param {dict} camera_param - camera parameter, cosisting of focal length and baseline 

    @return {Frame class} a Frame consists of kps, descs, and 3d coordinate
    """

    # Create an ORB Object to Extract Keypoints
    orb = cv2.ORB_create(edgeThreshold=51)

    # Find the Keypoints of Left and Right Images
    left_keypoints, left_desc = orb.detectAndCompute(left_img, None) 
    right_keypoints, right_desc = orb.detectAndCompute(right_img, None)

    # Convert keypoints to ndarray
    left_keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in left_keypoints])
    right_keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in right_keypoints])

    # Match the Left and Right Keypoints
    matches = feature_matching(left_keypoints, left_desc, right_keypoints, right_desc)

    # Create Frame Object
    frame_fps = FeaturePoints()
    frame_fps.left_pts = np.empty([0, 2])
    frame_fps.left_descriptors = np.empty([0, left_desc.shape[1]])
    frame_fps.right_pts = np.empty([0, 2])
    frame_fps.right_descriptor = np.empty([0, right_desc.shape[1]])
    frame_fps.pt3ds = np.empty([0, 3])

    # For all Matches
    for m in matches:

        # Get the Indices of Matches
        left_index = m.queryIdx
        right_index = m.trainIdx

        # Compute Feature Points and Disparity
        if (abs(left_keypoints[left_index][1] - right_keypoints[right_index][1]) > 1e-9):
            continue
        
        # Save the Coordinates of Left Image of Feature point
        frame_fps.left_pts = np.vstack([frame_fps.left_pts, [left_keypoints[left_index][0], left_keypoints[left_index][1]]])
        frame_fps.left_descriptors = np.vstack([frame_fps.left_descriptors, left_desc[left_index]])
        
        # Save the Coordinates of Right Image of Feature point
        frame_fps.right_pts = np.vstack([frame_fps.right_pts, [right_keypoints[right_index][0], right_keypoints[right_index][1]]])
        frame_fps.right_descriptor = np.vstack([frame_fps.right_descriptor, right_desc[right_index]])

        # Compute 3D coordinate
        neg_disparity = right_keypoints[right_index][0] - left_keypoints[left_index][0]
        z = camera_param['baseline'] * camera_param['focal_length'] / neg_disparity
        x = 0
        y = 0
        frame_fps.pt3ds = np.vstack([frame_fps.pt3ds, [x, y, z]])

    frame_fps.num_fp = int(frame_fps.left_pts.shape[0])
    print('Number of matches', frame_fps.num_fp)

    return frame_fps

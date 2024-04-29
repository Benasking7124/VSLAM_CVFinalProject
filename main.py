# Import Necessary Modules
from read_camera_param import ReadCameraParam
from perform_yolo import PerformYolo
from feature_extraction import FeatureExtraction
from feature_extraction import FeatureExtraction
from filter_feature_points import FilterFeaturePoints
from frame_matching import FrameMatching
from pose_estimator import PoseEstimator
from bounding_box_association import BoundingBoxAssociation
from display_images import DisplayImages
from draw_trajectory import DrawTrajectory

# Import Necessary Libraries
import cv2
import os
import numpy as np
import time


# Define Dataset Folder
dataset = './Dataset_3'

# Define Main Function
if __name__ == "__main__":

    # Read Calib File
    camera_param = ReadCameraParam(dataset + '/calib.txt')
    T_true_path = dataset + '/true_T.txt'
    T_true = np.loadtxt(T_true_path, dtype=np.float64)

    # Get the Folders for Left & Right Stereo Images
    left_images_folder = dataset + '/Left_Images/'
    right_images_folder = dataset + '/Right_Images/'

    # Get the Images Path list
    left_images = sorted(os.listdir(left_images_folder))
    right_images = sorted(os.listdir(right_images_folder))

    # Get the Path of Images
    left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images][-5:]
    right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images][-5:]

    # Read the First Frame of Left and Right Images
    left_image = cv2.imread(left_images[0])
    right_image = cv2.imread(right_images[0])
    previous_feature_points = FeatureExtraction(left_image, right_image, camera_param)

    Transformation_list = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
    T_pre = np.identity(4)

    # For the Left and Right Images Dataset
    for ind in range(1, len(left_images)):

        ####################### Preprocess the Images #######################
        # Read the Images
        left_image = cv2.imread(left_images[ind])
        right_image = cv2.imread(right_images[ind])

        ########################### Perform YOLO on Both Images ########################
        left_boxes, right_boxes = PerformYolo(left_image, right_image)
        
        ############## Extract FeaturePoints from Both Images ##############
        # camera_param below is for Kitti dataset (Dataset_1, Dataset_3 not for Dataset_2)
        feature_points = FeatureExtraction(left_image, right_image, camera_param)

        ######################## Remove Static Features using Depth map ################
        static_feature_points, dynamic_feature_points = FilterFeaturePoints(left_boxes, right_boxes, feature_points, num_clusters = 2)

        ###################### Perform Bounding Box Association ########################
        #associated_bounding_boxes = BoundingBoxAssociation(left_boxes, right_boxes, dynamic_feature_points)

        ############################## Display both the Images #########################
        # DisplayImages(left_image, right_image, left_boxes, right_boxes, static_feature_points, dynamic_feature_points)

        ############################## Match Feature Between Frames #########################
        paired_static_features = FrameMatching(previous_feature_points, feature_points)
        previous_feature_points = feature_points

        ############# Compute Transformation matrix of Camera onto Next Frame ###############
        # Compute the Reprojection Error
        pe = PoseEstimator(paired_static_features, camera_param['left_projection'], Transformation_list[ind - 1])
        
        # Minimise the Reprojection Error
        T_current = pe.minimize_error().reshape(4, 4)
        
        # Compute the Transformation of Current frame with respect to Previous frame
        T_current = T_pre @ T_current
        T_pre = T_current
        
        # Stack the Transformation matrices
        Transformation_list = np.vstack([Transformation_list, T_current[0:3, :].flatten()])

    # Draw the Final Trajectory
    DrawTrajectory(Transformation_list)
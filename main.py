# Import Necessary Modules
from read_camera_param import ReadCameraParam
from perform_yolo import PerformYolo
from feature_extraction import FeatureExtraction
from feature_extraction import FeatureExtraction
from filter_feature_points import FilterFeaturePoints
from frame_matching import FrameMatching
from compute_reproj_error import ComputeReprojError
from bounding_box_association import BoundingBoxAssociation
from display_images import DisplayImages


# Import Necessary Libraries
import cv2
import os
import numpy as np
import time


# Define Main Function
if __name__ == "__main__":

    # Read Calib File
    camera_param = ReadCameraParam('./Dataset_1/calib.txt')

    # Get the Folders for Left & Right Stereo Images
    left_images_folder = 'Dataset_1/Left_Images/'
    right_images_folder = 'Dataset_1/Right_Images/'

    # Get the Images Path list
    left_images = os.listdir(left_images_folder)
    right_images = os.listdir(right_images_folder)

    # Get the Path of Images
    left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images]
    right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images]

    # Read the First Frame of Left and Right Images
    left_image = cv2.imread(left_images[0])
    right_image = cv2.imread(right_images[0])
    previous_feature_points = FeatureExtraction(left_image, right_image, camera_param)

    T_previous = np.identity(4)
    # For the Left and Right Images Dataset
    for ind in range(1, len(left_images)):

        ####################### Preprocess the Images #######################
        # Read the Images
        left_image = cv2.imread(left_images[ind])
        right_image = cv2.imread(right_images[ind])

        # Resize Images
        left_image = cv2.resize(left_image, (650, 350))
        right_image = cv2.resize(right_image, (650, 350))

        ########################### Perform YOLO on Both Images ########################
        left_boxes, right_boxes = PerformYolo(left_image, right_image)
        
        ############## Extract FeaturePoints from Both Images ##############
        # camera_param below is for Kitti dataset (Dataset_1, Dataset_3 not for Dataset_2)
        feature_points = FeatureExtraction(left_image, right_image, camera_param)

        ######################## Remove Static Features using Depth map ################
        static_feature_points, dynamic_feature_points = FilterFeaturePoints(feature_points, num_clusters = 2)

        # ###################### Perform Bounding Box Association ########################
        #associated_bounding_boxes = BoundingBoxAssociation(left_boxes, right_boxes, dynamic_feature_points)

        # ############################## Display both the Images #########################
        #DisplayImages(left_image, right_image, static_feature_points, dynamic_feature_points, associated_bounding_boxes)

        # ############################## Match Feature Between Frames #########################
        paired_static_features = FrameMatching(previous_feature_points, feature_points)
        previous_feature_points = feature_points

        # ############################## Compute Reprojection Error #########################
        Reprojection_error = ComputeReprojError(paired_static_features, camera_param, T_previous, T_current)
        time.sleep(10)
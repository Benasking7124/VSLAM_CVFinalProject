# Import Necessary Modules
from perform_yolo import PerformYolo
from feature_extraction import FeatureExtraction
from bounding_box_association import BoundingBoxAssociation
from filter_feature_points import FilterFeaturePoints
from display_images import DisplayImages

# Import Necessary Libraries
from ultralytics import YOLO
import cv2
import time
import os


# Define Main Function
if __name__ == "__main__":

    # Get the Folders for Left & Right Stereo Images
    left_images_folder = 'Dataset_2/Left_Images/'
    right_images_folder = 'Dataset_2/Right_Images/'
    depth_maps_folder = 'Dataset_2/Depth_Maps/'

    # Get the Images Path list
    left_images = os.listdir(left_images_folder)
    right_images = os.listdir(right_images_folder)
    depth_maps = os.listdir(depth_maps_folder)

    # Get the Path of Images
    left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images]
    right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images]
    depth_maps = [os.path.abspath(depth_maps_folder + '/' + depth_map) for depth_map in depth_maps]
    
    # Load the YOLOv8 Pretrained Model
    model = YOLO('yolov8n.pt')

    # For the Left and Right Images Dataset
    for ind in range(len(left_images)):

        ####################### Preprocess the Images #######################

        # Read the Images
        left_image = cv2.imread(left_images[ind])
        right_image = cv2.imread(right_images[ind])
        depth_map = cv2.imread(depth_maps[ind], cv2.IMREAD_ANYDEPTH)

        # Resize the Images
        left_image = cv2.resize(left_image, [650, 350])
        right_image = cv2.resize(right_image, [650, 350])
        depth_map = cv2.resize(depth_map, [650, 350])

        ########################### Perform YOLO on Both Images ########################
        left_boxes, right_boxes = PerformYolo(model, left_image, right_image)
        
        ############## Extract FeaturePoints & Disparity from Both Images ##############
        feature_points, disparity = FeatureExtraction(left_image, right_image)

        ######################## Remove Static Features using Depth map ################
        filtered_feature_points = FilterFeaturePoints(feature_points, depth_map, depth_threshold = 1000)

        ###################### Perform Bounding Box Association ########################
        BoundingBoxAssociation(left_image, right_image, left_boxes, right_boxes, filtered_feature_points)

        ############################## Display both the Images #########################
        DisplayImages(left_image, right_image, left_boxes, right_boxes, filtered_feature_points)
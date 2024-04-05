# Import Necessary Libraries
from perform_yolo import perform_yolo_on_images
from feature_extraction import featureExtraction
from bounding_box_association import BoundingBoxAssociation
from filter_featurepoints import filter_featurepointsbydepth
from ultralytics import YOLO
import cv2
import time
import os


if __name__ == "__main__":

    # Get the Folders for Left & Right Stereo Images
    left_images_folder = 'Dataset_2/Left_Images/'
    right_images_folder = 'Dataset_2/Right_Images/'

    # Get the Images Path list
    left_images = os.listdir(left_images_folder)
    right_images = os.listdir(right_images_folder)

    # Get the Path of Images
    left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images]
    right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images]

    # Load the YOLOv8 Pretrained Model
    model = YOLO('yolov8n.pt')

    # For the Left and Right Images Dataset
    for ind in range(len(left_images)):

        ####################### Preprocess the Images #######################

        # Read the Images
        left_image = cv2.imread(left_images[ind])
        right_image = cv2.imread(right_images[ind])

        # Resize the Images
        left_image = cv2.resize(left_image, [650, 350])
        right_image = cv2.resize(right_image, [650, 350])

        ########################### Perform YOLO on Both Images ########################
        left_boxes, right_boxes = perform_yolo_on_images(model, left_image, right_image)
        
        ############## Extract FeaturePoints & Disparity from Both Images ##############
        featurePoints, disparity = featureExtraction(left_image, right_image)

        ######################## Remove Static using Depth map #################
        filteredFeaturePoints = filter_featurepointsbydepth(featurePoints,left_image, right_image)
        ###################### Perform Bounding Box Association ########################
        BoundingBoxAssociation(left_boxes, right_boxes, filteredFeaturePoints)
        
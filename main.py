# Import Necessary Libraries
from perform_yolo import perform_yolo_on_images
from feature_extraction import featureExtraction
from ultralytics import YOLO
import cv2
import os


if __name__ == "__main__":

    # Define the Scenario Number
    scenario = '/Scenario_1/'

    # Get the Folders for Left & Right Stereo Images and Depth Maps
    left_images_folder = 'dataset/image_1/'
    right_images_folder = 'dataset/image_0/'
    # depth_images_folder = os.path.abspath(os.getcwd() + '../../Dataset/Depth_Maps/' + scenario)

    # Get the Images Path list from the Scenarios
    left_images = os.listdir(left_images_folder)
    left_images.sort(key=lambda x: int(x.split('.')[0]))
    right_images = os.listdir(right_images_folder)
    right_images.sort(key=lambda x: int(x.split('.')[0]))
    # depth_images = os.listdir(depth_images_folder)

    # Get the Path of Images
    left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images]
    right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images]
    # depth_images = [os.path.abspath(depth_images_folder + '/' + depth_image) for depth_image in depth_images]

    # Load the YOLOv8 Pretrained Model
    model = YOLO('yolov8n.pt')

    # Create a List of Necessary Detections Class names
    required_detections = ["person", "bicycle", "car", "bus", "truck"]

    # For the Left and Right Images Dataset
    for ind in range(len(left_images)):

        ####################### Preprocess the Images #######################

        # Read the Images
        left_image = cv2.imread(left_images[ind])
        right_image = cv2.imread(right_images[ind])

        # Resize the Images
        left_image = cv2.resize(left_image, [650, 350])
        right_image = cv2.resize(right_image, [650, 350])

        #################### Perform YOLO on Both Images ###################
        # Perform YOLO on Both Images
        boudingBoxes = perform_yolo_on_images(model, left_image, right_image, required_detections)
        
        #################### Extract FeaturePoints from Both Images ####################
        featurePoints = featureExtraction(left_image, right_image)
        
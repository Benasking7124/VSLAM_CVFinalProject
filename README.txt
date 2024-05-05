# Computer Vision Final Project
Our report, slides, and all of our code is included within this zip file.


# Documents:
-Computer_Vision_Final_Presentation_Reimplementation_of_Visual_SLAM_in_Dynamic_Environments_Based_on_Object_Detection_and_Scene_Flow.pdf: Presentation
- Computer_Vision_Final_Report_Reimplementation_of_Visual_SLAM_in_Dynamic_Environments_Based_on_Object_Detection_and_Scene_Flow.pdf: Final report


# Precomputed Results:
- Result: Saved graphs of Yaw, Trajectory for all Datasets


# Program Files
- landmark_recognition.ipynb: This file contains the data augmentation and training procedures
- predict.ipynb: This file can be used to perform predictions on the test set

- data_download_and_DELG_testing.ipynb: 	Notebook for downloading data in Google collab and attempting to download
							and install DELG. Does not work. Loosely reference 
- evaluation.py: 	Runs all of the evaluation we performed for our model. Requires a subdirectory test_images_model with
			all of the testing images to produce the "best" and "worst" predicted image class pictures. Generates
			all presented visualizations in the report for the evaluation


# Parameter Files:
- yolov8n.pt: Pretrained YOLOv8n Model

# Data Files
These files describe the modified data set we used for our project (a subset of KITTI)
- Dataset_00, Dataset_02, Dataset_05, Dataset_08, Dataset_10: 200 Stereo images, calib.txt having Camera Calibration Parameters, times.txt having Timestamp of every Image frame, true_T.txt having True Transformation Matrices
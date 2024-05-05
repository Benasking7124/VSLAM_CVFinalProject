# Computer Vision Final Project
Our report, slides, and all of our code is included within this zip file.

Pre-trained parameters are provided (model_dense.h5 and model_dense2.h5)
are pretrained and provided for performing predictions on the testing set.


# Documents:
- Computer_Vision_Final_Presentation_Landmark_Recognition.pdf: Presentation
- Computer_Vision_Final_Report_Landmark_Recognition.pdf: Final report
- DELF_and_DELG_Installation.pdf: Rough guide for installing DELF and DELG and recovering Oxford results


# Precomputed Results:
- test_result_dense.csv: Useful for running the evaluation program


# Program Files
- landmark_recognition.ipynb: This file contains the data augmentation and training procedures
- predict.ipynb: This file can be used to perform predictions on the test set

- data_download_and_DELG_testing.ipynb: 	Notebook for downloading data in Google collab and attempting to download
							and install DELG. Does not work. Loosely reference DELG_and_DELG_Installation.pdf
							for instructions on recovering the Oxford results

- evaluation.py: 	Runs all of the evaluation we performed for our model. Requires a subdirectory test_images_model with
			all of the testing images to produce the "best" and "worst" predicted image class pictures. Generates
			all presented visualizations in the report for the evaluation


# Parameter Files:
- model_dense.h5: Earlier checkpoint, presented in presentation
- model_dense2.h5: Final checkpoint

# Data Files
These files describe the modified data set we used for our project (a subset of GLDv2).
- train_clean.csv: The training split of the training data
- test_clean.csv: The  the test set

# Data
- Training Data: https://drive.google.com/drive/folders/11YnNvM7lZAoqfcU3dxXGMcrxL2t8c0Kx?usp=sharing
- Testing Data: https://drive.google.com/drive/folders/1SYKhn1JMYJCwbD_WpzuahJRgbPAGT9Pg?usp=sharing
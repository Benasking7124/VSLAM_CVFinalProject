# Import Necessary Modules
from feature_extraction import FeatureExtraction
from read_camera_param import read_camera_param

# Import Necessary Libraries
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def draw_features_on_image(left_image, right_image, left_pts, right_pts):
        
        radius = 3
        color = (0, 255, 0)
        thickness = 1
        font_scale = 0.5

        for i, (x, y) in enumerate(left_pts):
            # Draw a circle at the feature point
            cv2.circle(left_image, (int(x), int(y)), radius, color, thickness)
        

        for i, (x, y) in enumerate(right_pts):
            # Draw a circle at the feature point
            cv2.circle(right_image, (int(x), int(y)), radius, color, thickness)

        fig = plt.figure()

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(left_image)
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(right_image)
        
        plt.show()



# Define Main Function
if __name__ == "__main__":

    # Get the Folders for Left & Right Stereo Images
    left_images_folder = 'Dataset_3/Left_Images/'
    right_images_folder = 'Dataset_3/Right_Images/'

    # Get the Images Path list
    left_images = os.listdir(left_images_folder)
    right_images = os.listdir(right_images_folder)

    # Sort Imgae Paths by Name
    left_images = sorted(left_images, key=lambda x:int(x.split('.')[0]))
    right_images = sorted(right_images, key=lambda x:int(x.split('.')[0]))

    # Get the Path of Images
    left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images]
    right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images]
    
    # Get True T from the Given File
    # true_T_list = []

    # with open('./Dataset_3/true_T.txt', 'r') as file:
    #     for line in file:
    #         true_T_list.append(line.strip()) 
    
    file_path = './Dataset_3/true_T.txt'
    true_T = np.loadtxt(file_path, dtype=np.float64)


    # For the Left and Right Images Dataset
    for ind in range(len(left_images)):

        ####################### Preprocess the Images #######################

        # Read the Images
        left_image = cv2.imread(left_images[ind])
        right_image = cv2.imread(right_images[ind])

        # Resize the Images
        left_image = cv2.resize(left_image, [650, 350])
        right_image = cv2.resize(right_image, [650, 350])

        ############## Extract FeaturePoints & Disparity from Both Images ##############
        # camera_param below is for Kitti dataset (Dataset_1, Dataset_3 not for Dataset_2)
        camera_param = read_camera_param('./Dataset_1/calib.txt')
        feature_points = FeatureExtraction(left_image, right_image, camera_param)

        # Draw Feature Points on Images
        # draw_features_on_image(left_image, right_image, feature_points.left_pts, feature_points.right_pts)

        predicted_list = []

        for pt3d in feature_points.pt3ds:
            temp = true_T[ind][:9].reshape(3,3)@pt3d
            temp_2 = np.linalg.inv(true_T[ind+1][:9].reshape(3,3))@temp
        

            f = camera_param['focal_length']  # Focal length
            M = np.array([
                [f, 0, 650/2, 0],
                [0, f, 350/2, 0],
                [0, 0, 1, 0]
            ])


            temp_3 = np.hstack((temp_2.reshape(1,3), np.ones((1,1))))
            projected_homogeneous = M@temp_3.T


            x_prime = projected_homogeneous[0, :] / projected_homogeneous[2, :]
            y_prime = projected_homogeneous[1, :] / projected_homogeneous[2, :]

            print(x_prime, y_prime)

      

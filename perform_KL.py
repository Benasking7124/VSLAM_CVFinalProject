# Import Necessary Modules
from feature_extraction import FeatureExtraction
from read_camera_param import ReadCameraParam
from frame_matching import FrameMatching
from scipy.stats import entropy
from perform_yolo import PerformYolo


# Import Necessary Libraries
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def draw_features_on_image(img_1, img_2, img1_pts, img2_pts, pic_title):
        
        radius = 5
        color = (0, 255, 0)
        thickness = 1

        img_1_copy = img_1.copy()
        img_2_copy = img_2.copy()


        for i, (x, y) in enumerate(img1_pts):
            cv2.circle(img_1_copy, (int(x), int(y)), radius, color, thickness)

        for i, (x, y) in enumerate(img2_pts):
            cv2.circle(img_2_copy, (int(x), int(y)), radius, color, thickness)


        fig = plt.figure()

        plt.title(pic_title)
        plt.imshow(np.vstack((img_1_copy, img_2_copy)))

        for (x1, y1), (x2, y2) in zip(img1_pts, img2_pts):
            plt.plot([x1, x2], [y1, y2+img_1_copy.shape[0]], 'b', linewidth=0.5)
        

        plt.show()

def visualize_KL(img_1, img_2, img1_pts, img2_pts, kl_values, left_boxes, right_boxes, pic_title):
        
        radius = 2
        color = (0, 255, 0)
        thickness = 2

        img_1_copy = img_1.copy()
        img_2_copy = img_2.copy()


        for i, (x, y) in enumerate(img1_pts):
            cv2.circle(img_1_copy, (int(x), int(y)), radius, color, thickness)

        for i, (x1, y1, x2, y2) in enumerate(left_boxes):
            cv2.rectangle(img_1_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)

        for i, (x, y) in enumerate(img2_pts):
            cv2.circle(img_2_copy, (int(x), int(y)), radius, color, thickness)

        for i, (x1, y1, x2, y2) in enumerate(right_boxes):
            cv2.rectangle(img_2_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)


        fig = plt.figure()

        plt.title(pic_title)
        plt.imshow(np.vstack((img_1_copy, img_2_copy)))

        for (x1, y1), (x2, y2), one_kl_value in zip(img1_pts, img2_pts, kl_values):
            plt.plot([x1, x2], [y1, y2+img_1_copy.shape[0]], 'b', linewidth=0.5)
            plt.text(x1, y1, one_kl_value, fontsize=8, color='red', rotation=45)
        

        plt.show()

def normlize_descriptors(desc):
    norm_desc = np.abs(desc)
    total = np.sum(norm_desc)
    if total>0:
        norm_desc /= total

    return norm_desc

def kl_divergence_scipy(P, Q, epsilon=1e-10):

    P = np.array(P) + epsilon
    Q = np.array(Q) + epsilon
    return entropy(P, Q)

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
    
    
    file_path = './Dataset_3/true_T.txt'
    true_T = np.loadtxt(file_path, dtype=np.float64)


    # For the Left and Right Images Dataset
    for ind in range(len(left_images)):


        # Read the Images
        left_image = cv2.imread(left_images[ind])
        right_image = cv2.imread(right_images[ind])


        next_left_image = cv2.imread(left_images[ind+1])
        next_right_image = cv2.imread(right_images[ind+1])

        left_boxes, right_boxes = PerformYolo(next_left_image, next_right_image)

        camera_param = ReadCameraParam('./Dataset_3/calib.txt')
        feature_points = FeatureExtraction(left_image, right_image, camera_param)
        next_feature_points = FeatureExtraction(next_left_image, next_right_image, camera_param)


        paired_features = FrameMatching(feature_points, next_feature_points)


        predicted_list = []

        for ind, pt3d in enumerate(paired_features[:, 2:]):
            
            temp = np.vstack((true_T[ind].reshape(3, 4), [0, 0, 0, 1])) @ np.hstack((pt3d, [1]))
            temp_2 = np.linalg.inv(np.vstack((true_T[ind+1].reshape(3, 4), [0, 0, 0, 1])))@temp
        
            M = camera_param['left_projection']

            projected_homogeneous = M@temp_2

            x_prime = projected_homogeneous[0] / projected_homogeneous[2]
            y_prime = projected_homogeneous[1] / projected_homogeneous[2]

            if np.isnan(x_prime)==False or np.isnan(y_prime)==False:
                predicted_list.append([x_prime, y_prime])

        predicted_list = np.array(predicted_list)


        current_2ds = []
        previous_2ds = []

        for ind, one_pair in enumerate(paired_features):
            
            previous_2d = M@np.hstack((one_pair[2:], [1]))

            x_prime = previous_2d[0] / previous_2d[2]
            y_prime = previous_2d[1] / previous_2d[2]

            if np.isnan(x_prime)==False or np.isnan(y_prime)==False:
                previous_2ds.append([x_prime, y_prime])
                current_2ds.append(one_pair[:2])


        # Visualization to check feature matching between time frames, predicted points vs observed points
        # draw_features_on_image(left_image, next_left_image, previous_2ds, predicted_list, 'Observed Fpts in T0 & Predicted Fpts in T1')
        # draw_features_on_image(left_image, next_left_image, previous_2ds, current_2ds, 'Observed Fpts Matching between T0 and T1')
        # draw_features_on_image(next_left_image, next_left_image, predicted_list, current_2ds, 'Compare Predicted Fpts in T1 & Observed Fpts in T1')


        orb = cv2.ORB_create()
        
        keypoints = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=50) for pt in predicted_list]
        __, preds_descriptors = orb.compute(left_image, keypoints)
        preds_descriptors = np.array(preds_descriptors, dtype=np.float64)

        keypoints = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=50) for pt in current_2ds]
        __, observed_descriptors = orb.compute(left_image, keypoints)     
        observed_descriptors = np.array(observed_descriptors, dtype=np.float64)

        kl_values = []

        for one_pred_desc, one_observed_desc in zip(preds_descriptors, observed_descriptors):

            norm_one_pred_desc = normlize_descriptors(one_pred_desc)
            norm_one_observed_desc = normlize_descriptors(one_observed_desc)

            one_entropy = kl_divergence_scipy(norm_one_pred_desc, norm_one_observed_desc)

            kl_values.append(round(one_entropy, 3))
        
        
        visualize_KL(next_left_image, next_left_image, predicted_list, current_2ds, 
                     kl_values, left_boxes, left_boxes, 'Compare Predicted Fpts in T1 & Observed Fpts in T1')


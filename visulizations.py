# Import Necessary Modules
from scipy.stats import entropy

# Import Necessary Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time



def draw_features_on_image_vertical(img_1, img_2, img1_pts, img2_pts, pic_title):
        
        radius = 5
        color = (0, 255, 0)
        thickness = 2

        img_1_copy = img_1.copy()
        img_2_copy = img_2.copy()


        for i, (x, y) in enumerate(img1_pts):
            cv2.circle(img_1_copy, (int(x), int(y)), radius, color, thickness)

        for i, (x, y) in enumerate(img2_pts):
            cv2.circle(img_2_copy, (int(x), int(y)), radius, color, thickness)


        fig = plt.figure(figsize=(10, 5))

        plt.title(pic_title)
        plt.imshow(np.vstack((img_1_copy, img_2_copy)))

        for (x1, y1), (x2, y2) in zip(img1_pts, img2_pts):
            plt.plot([x1, x2], [y1, y2+img_1_copy.shape[0]], 'b', linewidth=1)
        

        return fig





def draw_features_on_image_together(img_1, img_2, img1_pts, img2_pts, pic_title):
    
    radius = 5
    thickness = 2

    img_1_copy = img_1.copy()


    for i, (x, y) in enumerate(img1_pts):
        cv2.circle(img_1_copy, (int(x), int(y)), radius, (0, 255, 0), thickness)

    for i, (x, y) in enumerate(img2_pts):
        cv2.circle(img_1_copy, (int(x), int(y)), radius, (255, 0, 0), thickness)


    fig = plt.figure()
    plt.title(pic_title)
    plt.imshow(img_1_copy)
    plt.title(pic_title)

    # for (x1, y1), (x2, y2) in zip(img1_pts, img2_pts):
    #     plt.plot([x1, x2], [y1, y2+img_1_copy.shape[0]], 'b', linewidth=0.5)
    

    plt.show()





def visualize_KL(img_1, img_2, img1_pts, img2_pts, kl_values, left_boxes, right_boxes, font_size, pic_title):
        
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
            plt.text(x1, y1, one_kl_value, fontsize=font_size, color='red', rotation=45)
        

        plt.show()


import cv2
import os
import matplotlib.pyplot as plt
from feature_extraction import FeatureExtraction

if __name__ == "__main__":
    left_images_folder = 'Dataset_1/Left_Images/'
    right_images_folder = 'Dataset_1/Right_Images/'
    # disparity_folder = 'Dataset_2/Dispartiy_Maps'

    left_images = os.listdir(left_images_folder)
    right_images = os.listdir(right_images_folder)
    # disparity_maps = os.listdir(disparity_folder)

    left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images]
    right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images]
    # disparity_maps = [os.path.abspath(disparity_folder + '/' + depth_map) for depth_map in disparity_maps]

    left_image = cv2.imread(left_images[2])
    right_image = cv2.imread(right_images[2])
    # disparity_image = cv2.imread(disparity_maps[0])

    feature_points = FeatureExtraction(left_image, right_image)
    # It seems like for dataset2, the left image is actually the right image
    # feature_points = FeatureExtraction(right_image, left_image)

    # print(feature_points)

    fp_sorted = sorted(feature_points, key=lambda x: x.disparity)

    for fp in fp_sorted:
        lu = int(fp.left_pt[0])
        lv = int(fp.left_pt[1])
        cv2.circle(left_image, (lu, lv), radius=5, color=(255, 0, 0), thickness=-1)

        ru = int(fp.right_pt[0])
        rv = int(fp.right_pt[1])
        cv2.circle(right_image, (ru, rv), radius=5, color=(255, 0, 0), thickness=-1)

        fig, axes = plt.subplots(1, 2, figsize=(50, 10))
        axes[0].imshow(left_image)
        axes[0].set_title('Left 1')
        axes[1].imshow(right_image)
        axes[1].set_title('Right 2')
        plt.tight_layout()
        plt.text(0, 0, f'({fp.disparity})')
        plt.show()
        cv2.circle(left_image, (lu, lv), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(right_image, (ru, rv), radius=5, color=(0, 0, 255), thickness=-1)

        # depth_pos = [int((lu + ru) / 2), int((lv + rv) / 2)]
        # print(f'({fp.disparity}, {disparity_image[depth_pos[1], depth_pos[0]]})')
    print("-----")
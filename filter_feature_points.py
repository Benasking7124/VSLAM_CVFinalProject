# Import Necessary Libraries
import numpy as np
import cv2


# Define a Function to Filter Feature Points by Depth
def FilterFeaturePoints(featurePoints, depth_map, depth_threshold):
    
    # Initialise List to Store Filtered Points
    filtered_feature_points = []
    
    # For every Feature Point
    for point in featurePoints:

        # Get Coordinates from both Images
        left_x, left_y, right_x, right_y = int(point[0]), int(point[1]), int(point[2]), int(point[3])
        
        # Get Depth and Compare with Threshold to Filter
        depth_pos = [int((left_x + right_x) / 2), int((left_y + right_y) / 2)]
        depth = depth_map[depth_pos[1], depth_pos[0]]
        if depth > depth_threshold:
            filtered_feature_points.append(point)
    
    # Return the Filtered Feature Points
    return np.array(filtered_feature_points)
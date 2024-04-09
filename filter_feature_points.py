# Import Necessary Libraries
import numpy as np


# Define a Function to Filter Feature Points by Depth
def FilterFeaturePoints(featurePoints, depth_map, depth_threshold):
    
    # Initialise List to Store Static & Dynamic Feature Points
    static_feature_points = []
    dynamic_feature_points = []
    
    # For every Feature Point
    for point in featurePoints:

        # Get Coordinates from both Images
        left_x, left_y, right_x, right_y = int(point[0]), int(point[1]), int(point[2]), int(point[3])
        
        # Get Depth and Compare with Threshold to Filter
        depth_pos = [int((left_x + right_x) / 2), int((left_y + right_y) / 2)]
        depth = depth_map[depth_pos[1], depth_pos[0]]
        if depth < depth_threshold:
            dynamic_feature_points.append([left_x, left_y, right_x, right_y, depth])
        else:
            static_feature_points.append([left_x, left_y, right_x, right_y, depth])
    
    # Convert into Numpy Array
    static_feature_points = np.array(static_feature_points)
    dynamic_feature_points = np.array(dynamic_feature_points)
    
    # Return the Static & Dynamic Feature Points
    return static_feature_points, dynamic_feature_points
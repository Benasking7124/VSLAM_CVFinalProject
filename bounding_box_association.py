
# Define a Function to Check if a Feature Point falls inside Bounding Box
def check_point_in_bbox(bbox, point):

    # Get the Coordinates and Points
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    x, y = point[0], point[1]

    # Check if Point falls inside Bounding Box
    if x > x1 and x < x2 and y > y1 and y < y2:
        return True
    else:
        return False
    

# Define a Function to Count Matching Features
def Count_Matching_Features(feature_points, left_features, right_features):
    
    # Convert feature_points to a set of tuples
    feature_points_set = {tuple(point) for point in feature_points}

    # Convert left_features and right_features to sets of tuples
    left_features_set = {tuple(feature[:2]) for feature in left_features}
    right_features_set = {tuple(feature[:2]) for feature in right_features}

    # Calculate the intersection of left_features_set and right_features_set
    matching_pairs = left_features_set & right_features_set

    # Initialize match count
    match_count = 0

    # Iterate over the matching pairs
    for left_feature in matching_pairs:
        left_x, left_y = left_feature

        # Iterate over feature_points_set to find matching pairs
        for right_feature in right_features_set:
            right_x, right_y = right_feature

            # Check if the matching pair exists in feature_points_set
            if (left_x, left_y, right_x, right_y) in feature_points_set:
                match_count += 1

    # Return the match count
    return match_count


# Define a Function to Associate Bounding Boxes between Left and Right images
def BoundingBoxAssociation(left_boxes, right_boxes, feature_points):

    # Initialise Empty Dictionary to store Objects in Left and Right Images
    objects_on_left_image = dict()
    objects_on_right_image = dict()

    # Store the Bounding Boxes for both Frames
    objects_on_left_image['Bounding_Boxes'] = dict()
    objects_on_left_image['Bounding_Boxes']['Coordinates'] = left_boxes
    objects_on_right_image['Bounding_Boxes'] = dict()
    objects_on_right_image['Bounding_Boxes']['Coordinates'] = right_boxes

    # Store the Feature Points for both Frames
    objects_on_left_image['Feature_Points'] = feature_points[:, 0: 2]
    objects_on_right_image['Feature_Points'] = feature_points[:, 2: 4]

    # Initialise a List to store Feature Points for Corresponding Bounding Boxes
    objects_on_left_image['Bounding_Boxes']['Feature_Points'] = []
    objects_on_right_image['Bounding_Boxes']['Feature_Points'] = []

    # Check every Bounding Box Coordinates in Left Image
    for bbox in objects_on_left_image['Bounding_Boxes']['Coordinates']:

        # Initialise List of Points
        points = []

        # Check every Feature Point in Left Image
        for point in objects_on_left_image['Feature_Points']:
            
            # Check if that Points falls inside Bounding box
            if check_point_in_bbox(bbox, point):
                points.append(point)
        
        # Store the Feature Points for that Bounding Box
        objects_on_left_image['Bounding_Boxes']['Feature_Points'].append(points)
    

    # Check every Bounding Box Coordinates in Right Image
    for bbox in objects_on_right_image['Bounding_Boxes']['Coordinates']:

        # Initialise List of Points
        points = []

        # Check every Feature Point in Right Image
        for point in objects_on_right_image['Feature_Points']:
            
            # Check if that Points falls inside Bounding box
            if check_point_in_bbox(bbox, point):
                points.append(point)
        
        # Store the Feature Points for that Bounding Box
        objects_on_right_image['Bounding_Boxes']['Feature_Points'].append(points)
    

    # Initialise List to store Associated Bounding Boxes
    associated_bounding_boxes = []

    # For every Bounding box in Left Image
    for left_bbox_ind in range(len(objects_on_left_image['Bounding_Boxes']['Coordinates'])):

        # Get the Feature Points and Initialise Matches Count for Features
        left_features = objects_on_left_image['Bounding_Boxes']['Feature_Points'][left_bbox_ind]
        matches = []

        # For every Bounding box in Right Image
        for right_bbox_ind in range(len(objects_on_right_image['Bounding_Boxes']['Coordinates'])):

            # Get the Feature Points
            right_features = objects_on_right_image['Bounding_Boxes']['Feature_Points'][right_bbox_ind]
            
            # Append the Matches count
            matches.append(Count_Matching_Features(feature_points, left_features, right_features))
        
        # Get the Highest Match count and its Index if count > 0
        max_match = max(matches)
        if max_match > 0:
            max_match_index = matches.index(max_match)

            # Append the Matching Bounding Boxes
            associated_bounding_boxes.append([objects_on_left_image['Bounding_Boxes']['Coordinates'][left_bbox_ind], objects_on_right_image['Bounding_Boxes']['Coordinates'][max_match_index]])
    
    # Return Associated Bounding Boxes
    return associated_bounding_boxes
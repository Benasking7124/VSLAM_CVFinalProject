class FeaturePoints:
    """
    Class for a collection of FeaturePoints
    """
    def __init__(self) -> None:

        self.num_fp = 0   # number of feature points
        
        # Left image frame
        self.left_pts = 0   # numpy array, shape (n, 2)
        self.left_descriptors = 0   # numpy array
        
        # Right image frame
        self.right_pts = 0
        self.right_descriptor = 0

        # Camera frame
        self.pt3ds = 0   # numpy array, shape (n, 3)
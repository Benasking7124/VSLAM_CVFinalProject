class FeaturePoint:
    def __init__(self) -> None:
        # Left image frame
        self.left_pt = ()   # tuple of x and y
        self.left_descriptor = 0   # numpy array
        # Right image frame
        self.right_pt = ()
        self.right_descriptor = 0

        # Camera frame
        self.disparity = 0
        self.pt3d = ()   # tuple of x, y, and z
        self.depth = 0

        self.is_dynamic = False
        self.is_moving = False
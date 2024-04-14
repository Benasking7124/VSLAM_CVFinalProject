# Create a Class for Feature Point
class FeaturePoint:

    # Initialise Class Object
    def __init__(self) -> None:

        # Initialise Coordinates and Descriptor for Left Image
        self.left_pt = ()
        self.left_descriptor = 0

        # Initialise Coordinates and Descriptor for Right Image
        self.right_pt = ()
        self.right_descriptor = 0

        # Initialise Depth for the Point
        self.depth = 0
        
        # Initialise World Coordinates of Feature Points
        self.pt3d = ()

        # Initialise Flag to store if Point is Dynamic
        self.is_dynamic = False
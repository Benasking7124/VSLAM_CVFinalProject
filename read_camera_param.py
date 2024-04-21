import numpy as np

# The functions are copy from https://github.com/utiasSTARS/pykitti.git

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def read_camera_param(filepath):
    """
    Read Camera Parameter from calib.txt and output focal length and baseline

    @param {str} filepath - path of calib.txt

    @return {dict} dictionary containing projection matrices, focal length, and baseline
    """

    # Create Camera Parameter Dictionary
    camera_param = {}
     
    # Read Calib File
    calib_data = read_calib_file(filepath)   # type: dict

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(calib_data['P0'], (3, 4))
    P_rect_10 = np.reshape(calib_data['P1'], (3, 4))

    # Extract Projection Matrices, Focal Length, and Baselie
    camera_param['left_projection'] = P_rect_00
    camera_param['right_projection'] = P_rect_10
    camera_param['focal_length'] = P_rect_00[0][0]
    camera_param['baseline'] = -P_rect_10[0, 3] / P_rect_10[0, 0]
    # camera_param['principle_point'] = (P_rect_00[0, 2], P_rect_00[1, 2])

    return camera_param
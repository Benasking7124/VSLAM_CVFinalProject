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

    @return {dict} a dictionary contains focal length and baseline
    """

    # Create Camera Parameter Dictionary
    camera_param = {}
     
    # Read Calib File
    calib_data = read_calib_file(filepath)   # type: dict

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(calib_data['P0'], (3, 4))
    P_rect_10 = np.reshape(calib_data['P1'], (3, 4))

    # Extract Focal Length
    camera_param['focal_length'] = P_rect_00[0][0]


    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    
    # Compute the velodyne to rectified camera coordinate transforms
    data = {}
    data['T_cam0_velo'] = np.reshape(calib_data['Tr'], (3, 4))
    data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
    data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
    
    # Compute the stereo baselines in meters by projecting the origin of
    # each camera frame into the velodyne frame and computing the distances
    # between them
    p_cam = np.array([0, 0, 0, 1])
    p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
    p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
    
    # Calculate Baseline
    camera_param['baseline'] = np.linalg.norm(p_velo1 - p_velo0)

    return camera_param
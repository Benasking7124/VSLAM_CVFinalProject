import numpy as np
import cv2

def compute_ate(gt_poses, est_poses):
    """
    Compute Absolute Trajectory Error (ATE) between ground truth poses and estimated poses.
    """

    absolute_errors = np.linalg.norm(gt_poses[:, 9:12] - est_poses[:, 9:12], axis=1)
    ate = np.mean(absolute_errors)
    return ate

def compute_rpe_t(gt_poses, est_poses):
    """
    Compute Relative Pose Error (RPE) between ground truth poses and estimated poses.
    """

    relative_errors = np.linalg.norm(gt_poses[1:, 9:12] - gt_poses[:-1, 9:12] - (est_poses[1:, 9:12] - est_poses[:-1, 9:12]), axis=1)
    rpe = np.mean(relative_errors)
    return rpe

def compute_rpe_R(gt_poses, est_poses):
    """
    Compute Relative Pose Error (RPE) between ground truth poses and estimated poses.
    """

    r_gt = np.empty([0, 3])
    r_est = np.empty([0, 3])

    for i in range(gt_poses.shape[0]):
        Ri_gt = gt_poses[i].reshape(3, 4)
        Ri_gt = Ri_gt[:3, :3]
        ri_gt, _ = cv2.Rodrigues(Ri_gt)
        r_gt = np.vstack([r_gt, ri_gt.flatten()])

        Ri_est = est_poses[i].reshape(3, 4)
        Ri_est = Ri_est[:3, :3]
        ri_est, _ = cv2.Rodrigues(Ri_est)
        r_est = np.vstack([r_est, ri_est.flatten()])

    relative_errors = np.linalg.norm(r_gt[1:] - r_gt[:-1] - (r_est[1:] - r_est[:-1]), axis=1)
    rpe = np.mean(relative_errors)
    return rpe
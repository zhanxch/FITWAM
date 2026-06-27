import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from .constants import (
    CAM_AXIS_TRANSFORM,
    GT_CAM_QUAT_XYZW,
    MAIN_CAM_QUAT_XYZW,
    MAIN_CAM_TRANS,
    MAIN_INTRINSICS_1280X720,
)


def isaac_lab_camera_frame_change() -> np.ndarray:
    cam_rotmat = R.from_quat(MAIN_CAM_QUAT_XYZW).as_matrix()
    gt_rotmat = R.from_quat(GT_CAM_QUAT_XYZW).as_matrix()
    return (gt_rotmat @ np.linalg.inv(cam_rotmat)).astype(np.float32)


def main_cam_transformation() -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = isaac_lab_camera_frame_change() @ R.from_quat(MAIN_CAM_QUAT_XYZW).as_matrix()
    transform[:3, -1] = MAIN_CAM_TRANS
    return transform


MAIN_CAM_TRANSFORMATION = main_cam_transformation()
INV_MAIN_CAM_TRANSFORMATION = np.linalg.inv(MAIN_CAM_TRANSFORMATION).astype(np.float32)


def homogeneous_coord(points: np.ndarray) -> np.ndarray:
    ones_shape = list(points.shape)
    ones_shape[-1] = 1
    return np.concatenate([points, np.ones(ones_shape, dtype=points.dtype)], axis=-1)


def to_pose(ee_pose: np.ndarray) -> np.ndarray:
    transforms = np.zeros((ee_pose.shape[0], 4, 4), dtype=np.float32)
    # EgoVLA_SIM stores quaternions as WXYZ; scipy expects XYZW.
    quat_xyzw = np.concatenate([ee_pose[:, 4:], ee_pose[:, 3:4]], axis=1)
    transforms[:, :3, :3] = R.from_quat(quat_xyzw).as_matrix()
    transforms[:, :3, -1] = ee_pose[:, :3]
    transforms[:, 3, 3] = 1.0
    return transforms


def to_cam_frame(points: np.ndarray) -> np.ndarray:
    cam_points = CAM_AXIS_TRANSFORM @ INV_MAIN_CAM_TRANSFORMATION @ homogeneous_coord(points)[..., np.newaxis]
    return cam_points[..., :3, 0].astype(np.float32)


def pose_to_cam_frame(poses: np.ndarray) -> np.ndarray:
    return (CAM_AXIS_TRANSFORM @ INV_MAIN_CAM_TRANSFORMATION @ poses.reshape(-1, 4, 4)).astype(np.float32)


def scaled_intrinsics(height: int, width: int) -> np.ndarray:
    intrinsics = MAIN_INTRINSICS_1280X720.copy()
    intrinsics[0] *= width / 1280.0
    intrinsics[1] *= height / 720.0
    return intrinsics


def project_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = points.reshape(-1, 3).astype(np.float32)
    projected, _ = cv2.projectPoints(
        points,
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        intrinsics,
        np.array([], dtype=np.float32),
    )
    return projected.reshape(-1, 2)


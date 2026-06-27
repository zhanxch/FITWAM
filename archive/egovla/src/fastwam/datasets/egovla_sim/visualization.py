import cv2
import numpy as np

from .constants import SIM_HAND_CONNECTIVITY
from .geometry import pose_to_cam_frame, project_points, scaled_intrinsics, to_cam_frame, to_pose


def draw_sim_hand(frame_rgb: np.ndarray, points_2d: np.ndarray, color: tuple[int, int, int]) -> None:
    if np.isnan(points_2d).any() or np.isinf(points_2d).any():
        return
    height, width = frame_rgb.shape[:2]
    for start, end in SIM_HAND_CONNECTIVITY:
        p0 = points_2d[start]
        p1 = points_2d[end]
        if (
            -width <= p0[0] <= 2 * width
            and -height <= p0[1] <= 2 * height
            and -width <= p1[0] <= 2 * width
            and -height <= p1[1] <= 2 * height
        ):
            cv2.line(
                frame_rgb,
                (int(round(p0[0])), int(round(p0[1]))),
                (int(round(p1[0])), int(round(p1[1]))),
                color,
                thickness=2,
            )
    for point in points_2d:
        if -width <= point[0] <= 2 * width and -height <= point[1] <= 2 * height:
            cv2.circle(frame_rgb, (int(round(point[0])), int(round(point[1]))), radius=3, color=color, thickness=-1)


def overlay_hand_pose(frames_rgb: np.ndarray, hand: dict[str, np.ndarray | None]) -> np.ndarray:
    required = ("left_ee_pose", "right_ee_pose", "left_finger_tip_pos", "right_finger_tip_pos")
    if any(hand[key] is None for key in required):
        return frames_rgb

    frames = frames_rgb.copy()
    height, width = frames.shape[1:3]
    intrinsics = scaled_intrinsics(height, width)

    left_ee_cam = pose_to_cam_frame(to_pose(np.asarray(hand["left_ee_pose"], dtype=np.float32)))[:, :3, -1]
    right_ee_cam = pose_to_cam_frame(to_pose(np.asarray(hand["right_ee_pose"], dtype=np.float32)))[:, :3, -1]
    left_tips_cam = to_cam_frame(np.asarray(hand["left_finger_tip_pos"], dtype=np.float32))
    right_tips_cam = to_cam_frame(np.asarray(hand["right_finger_tip_pos"], dtype=np.float32))

    for idx, frame in enumerate(frames):
        left_points = np.concatenate([left_ee_cam[idx : idx + 1], left_tips_cam[idx]], axis=0)
        right_points = np.concatenate([right_ee_cam[idx : idx + 1], right_tips_cam[idx]], axis=0)
        if np.all(left_points[:, 2] > 1e-6):
            draw_sim_hand(frame, project_points(left_points, intrinsics), (255, 0, 0))
        if np.all(right_points[:, 2] > 1e-6):
            draw_sim_hand(frame, project_points(right_points, intrinsics), (0, 255, 0))

    return frames


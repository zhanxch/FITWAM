from pathlib import Path

import numpy as np


VIDEO_KEY = "observation.images.camera_0"
CHUNKS_SIZE = 1000

EGOVLA_SIM_ROOT = Path("data/EgoVLA_SIM")

CAM_AXIS_TRANSFORM = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)
MAIN_CAM_TRANS = np.array([0.09, 0.0, 1.7], dtype=np.float32)
MAIN_CAM_QUAT_XYZW = np.array([0.24184, -0.24184, -0.664464, 0.66446], dtype=np.float32)
GT_CAM_QUAT_XYZW = np.array([0.0, 0.42261826174069944, 0.0, 0.9063077870366499], dtype=np.float32)
MAIN_INTRINSICS_1280X720 = np.array(
    [
        [488.6662, 0.0, 640.0],
        [0.0, 488.6662, 360.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
SIM_HAND_CONNECTIVITY = ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))

LONG_TASKS: tuple[tuple[str, str], ...] = (
    ("Insert-And-Unload-Cans", "Insert and unload cans."),
    ("Stack-Can-Into-Drawer", "Stack the can into the drawer."),
    ("Sort-Cans", "Sort the cans."),
    ("Unload-Cans", "Unload the cans."),
    ("Insert-Cans", "Insert the cans."),
)

SHORT_TASKS: tuple[tuple[str, str], ...] = (
    ("Close-Drawer", "Close the drawer."),
    ("Flip-Mug", "Flip the mug."),
    ("Open-Drawer", "Open the drawer."),
    ("Open-Laptop", "Open the laptop."),
    ("Pour-Balls", "Pour the balls."),
    ("Push-Box", "Push the box."),
    ("Stack-Can", "Stack the can."),
)


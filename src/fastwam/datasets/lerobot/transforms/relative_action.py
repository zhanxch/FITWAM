from typing import List, Dict

import torch
from .rotation import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)

# pose: position and quaternion in x, y, z, i, j, k, r
# mat: homogeneous transformation matrix in 4×4
# pos: position in x, y, z
# quat: quaternion in i, j, k, r


class RelativePoseTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys
    
    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._forward(action, state[..., -1:, :])
        
        return batch
        
    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._backward(action, state[..., -1:, :])

        return batch

    def _forward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._absolute_to_relative(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    def _backward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._relative_to_absolute(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    @staticmethod
    def _pose_to_matrix(pose: torch.Tensor):
        position = pose[..., 0: 3]
        quaternion = pose[..., [6, 3, 4, 5]] # (i j k r) to (r i j k)
        rotation = quaternion_to_matrix(quaternion)
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0: 3, 0: 3] = rotation
        matrix[..., 0: 3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: torch.Tensor):
        position = matrix[..., 0: 3, 3] / matrix[..., 3, 3][..., None]
        rotation = matrix[..., 0: 3, 0: 3]
        quaternion = matrix_to_quaternion(rotation)
        quaternion = quaternion[..., [1, 2, 3, 0]] # (r i j k) to (i j k r)
        pose = torch.cat([position, quaternion], dim=-1)
        return pose
    @staticmethod
    def _absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return torch.linalg.inv(base_pose_matrix) @ pose_matrix

    @staticmethod
    def _relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return base_pose_matrix @ pose_matrix
    

class RelativeJointTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] - batch["state"][k][..., :1, :]

        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] + batch["state"][k][..., :1, :]
        
        return batch


class RelativePoseRot6dTransform:
    """SE(3) relative pose transform for xyz+rot6d (9-dim) action keys.

    Mirrors GR00T's EndEffectorPose relative_chunking: converts each action
    pose to the frame of the LAST state pose via T_relative = inv(T_base) @ T.
    rot6d is recovered to a rotation matrix, the relative transform is computed
    in homogeneous coordinates, and the result is converted back to xyz+rot6d.
    Dimensionality is preserved (9 -> 9), satisfying FastWAM's
    action_state_transform shape assertion.
    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        if "action" not in batch:
            return batch
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._forward(action, state[..., -1:, :])
        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._backward(action, state[..., -1:, :])
        return batch

    def _forward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        assert pose.shape[-1] == 9, f"Pose shape must be (..., 9), but got {pose.shape}"
        assert base_pose.shape[-1] == 9, f"Base pose shape must be (..., 9), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._absolute_to_relative(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose

    def _backward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        assert pose.shape[-1] == 9, f"Pose shape must be (..., 9), but got {pose.shape}"
        assert base_pose.shape[-1] == 9, f"Base pose shape must be (..., 9), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._relative_to_absolute(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose

    @staticmethod
    def _pose_to_matrix(pose: torch.Tensor):
        # pose: (..., 9) = xyz(3) + rot6d(6)
        position = pose[..., 0:3]
        rot6d = pose[..., 3:9]
        rotation = rotation_6d_to_matrix(rot6d)
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0:3, 0:3] = rotation
        matrix[..., 0:3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: torch.Tensor):
        position = matrix[..., 0:3, 3] / matrix[..., 3:3+1, 3].clamp(min=1e-8)
        rotation = matrix[..., 0:3, 0:3]
        rot6d = matrix_to_rotation_6d(rotation)
        pose = torch.cat([position, rot6d], dim=-1)
        return pose

    @staticmethod
    def _absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return torch.linalg.inv(base_pose_matrix) @ pose_matrix

    @staticmethod
    def _relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return base_pose_matrix @ pose_matrix


class RelativeJointTransformLastFrame:
    """Joint relative transform using the LAST state frame as reference.

    GR00T uses state[-1] as the reference for joint relative actions
    (state_action_processor.py: reference_state = state[state_key][-1]).
    This differs from RelativeJointTransform which uses the first frame.
    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        if "action" not in batch:
            return batch
        for k in self.keys:
            batch["action"][k] = batch["action"][k] - batch["state"][k][..., -1:, :]
        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            batch["action"][k] = batch["action"][k] + batch["state"][k][..., -1:, :]
        return batch


def _slice_from_config(sl):
    """Convert Hydra/OmegaConf [start, stop] lists to slice objects."""
    if isinstance(sl, slice):
        return sl
    try:
        if len(sl) == 2:
            return slice(int(sl[0]), int(sl[1]))
    except TypeError:
        pass
    return sl


class Gr00tStyleRelativeTransform:
    """GR00T-style relative action for a single merged key with per-segment logic.

    Handles the spray_water 58-dim layout in a single key:
      left_eef:  dims [0:9]   = xyz(3) + rot6d(6)   -> SE(3) relative via inv(T_base)@T
      right_eef: dims [9:18]  = xyz(3) + rot6d(6)   -> SE(3) relative
      left_hand: dims [18:38] = 20 joints           -> joint relative (action - state[-1])
      right_hand:dims [38:58] = 20 joints           -> joint relative

    This mirrors GR00T's state_action_processor.py flow:
      - EEF keys with ActionRepresentation.RELATIVE -> EndEffectorPose relative_chunking
        (T_relative = inv(T_state[-1]) @ T_action, SE(3) correct)
      - joint keys with ActionRepresentation.RELATIVE -> JointPose relative (self - other)

    Dimensionality is preserved (58 -> 58), satisfying FastWAM's
    action_state_transform shape assertion.
    """

    def __init__(
        self,
        key: str = "default",
        eef_slices: Dict[str, List[int]] | None = None,
        joint_slices: Dict[str, List[int]] | None = None,
    ):
        self.key = key
        # Accept [start, stop] lists from Hydra configs and convert to slices.
        self.eef_slices = {
            name: _slice_from_config(sl)
            for name, sl in (eef_slices or {
                "left_eef": slice(0, 9),
                "right_eef": slice(9, 18),
            }).items()
        }
        self.joint_slices = {
            name: _slice_from_config(sl)
            for name, sl in (joint_slices or {
                "left_hand": slice(18, 38),
                "right_hand": slice(38, 58),
            }).items()
        }

    def forward(self, batch: Dict):
        if "action" not in batch:
            return batch
        action = batch["action"][self.key]
        state = batch["state"][self.key]
        base = state[..., -1:, :]
        out = action.clone()
        for name, sl in self.eef_slices.items():
            out[..., sl] = self._eef_relative(action[..., sl], base[..., sl])
        for name, sl in self.joint_slices.items():
            out[..., sl] = action[..., sl] - base[..., sl]
        batch["action"][self.key] = out
        return batch

    def backward(self, batch: Dict):
        action = batch["action"][self.key]
        state = batch["state"][self.key]
        base = state[..., -1:, :]
        out = action.clone()
        for name, sl in self.eef_slices.items():
            out[..., sl] = self._eef_absolute(action[..., sl], base[..., sl])
        for name, sl in self.joint_slices.items():
            out[..., sl] = action[..., sl] + base[..., sl]
        batch["action"][self.key] = out
        return batch

    @staticmethod
    def _eef_relative(pose: torch.Tensor, base_pose: torch.Tensor):
        assert pose.shape[-1] == 9
        pose_matrix = RelativePoseRot6dTransform._pose_to_matrix(pose)
        base_matrix = RelativePoseRot6dTransform._pose_to_matrix(base_pose)
        rel_matrix = torch.linalg.inv(base_matrix) @ pose_matrix
        return RelativePoseRot6dTransform._matrix_to_pose(rel_matrix)

    @staticmethod
    def _eef_absolute(pose: torch.Tensor, base_pose: torch.Tensor):
        assert pose.shape[-1] == 9
        pose_matrix = RelativePoseRot6dTransform._pose_to_matrix(pose)
        base_matrix = RelativePoseRot6dTransform._pose_to_matrix(base_pose)
        abs_matrix = base_matrix @ pose_matrix
        return RelativePoseRot6dTransform._matrix_to_pose(abs_matrix)

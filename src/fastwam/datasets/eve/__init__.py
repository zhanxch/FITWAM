"""EveRobot dataset adapters and sidecar utilities."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .manifest_dataset import EveManifestRobotVideoDataset


def __getattr__(name: str) -> Any:
    if name == "EveManifestRobotVideoDataset":
        from .manifest_dataset import EveManifestRobotVideoDataset

        return EveManifestRobotVideoDataset
    raise AttributeError(name)

__all__ = ["EveManifestRobotVideoDataset"]

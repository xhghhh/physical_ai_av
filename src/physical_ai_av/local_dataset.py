# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Local dataset loader for PhysicalAI-Autonomous-Vehicles dataset.

This module provides a PyTorch-style Dataset class for loading data from a local
directory structure instead of downloading from Hugging Face.

Expected directory structure:
    dataset_root/
        features.csv
        clip_index.parquet
        metadata/
            sensor_presence.parquet
            data_collection.parquet
        camera/
            camera_cross_left_120fov/
                camera_cross_left_120fov_chunk_0000.zip
                camera_cross_left_120fov_chunk_0001.zip
                ...
            camera_cross_right_120fov/
                ...
            camera_front_tele_30fov/
                ...
            camera_front_wide_120fov/
                ...
            camera_rear_left_70fov/
                ...
            camera_rear_right_70fov/
                ...
        labels/
            egomotion/
                egomotion.chunk_0000.zip
                egomotion.chunk_0001.zip
                ...
        lidar/ (optional)
            ...
        radar/ (optional)
            ...
"""

import io
import json
import logging
import pathlib
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from physical_ai_av import egomotion, video
from physical_ai_av.calibration import CameraIntrinsics, SensorExtrinsics, VehicleDimensions

logger = logging.getLogger(__name__)

# Camera name to index mapping
CAMERA_NAME_TO_INDEX = {
    "camera_cross_left_120fov": 0,  # often-used
    "camera_front_wide_120fov": 1,  # often-used
    "camera_cross_right_120fov": 2,  # often-used
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,  # often-used
}


class LocalPhysicalAIAVDataset(Dataset):
    """PyTorch Dataset for loading PhysicalAI-AV data from local directory.

    This dataset loads clips from a local directory structure and provides
    access to image frames and egomotion data in a format compatible with
    the original HuggingFace-based interface.

    Args:
        dataset_root: Path to the root directory of the dataset.
        clip_ids: Optional list of clip IDs to load. If None, all clips are loaded
            (or filtered by split if specified).
        split: Optional dataset split to filter by ("train", "val", or "test").
            If provided, only clips from this split will be loaded. Cannot be used
            together with clip_ids.
        cameras: List of camera names to load. Defaults to ["camera_front_wide_120fov"].
        t0_us: Initial timestamp offset in microseconds. Defaults to 0.
        dt_us: Time step between frames in microseconds. Defaults to 50_000 (20Hz).
        num_frames: Number of image frames to load per sample. Defaults to 4.
        num_history_steps: Number of history egomotion steps. Defaults to 16.
        num_future_steps: Number of future egomotion steps. Defaults to 64.

    Returns:
        A dictionary containing:
            - "image_frames": torch.Tensor of shape (N_cameras, num_frames, 3, H, W)
            - "camera_indices": torch.Tensor of shape (N_cameras,)
            - "ego_history_xyz": torch.Tensor of shape (1, 1, num_history_steps, 3)
            - "ego_history_rot": torch.Tensor of shape (1, 1, num_history_steps, 3, 3)
            - "ego_future_xyz": torch.Tensor of shape (1, 1, num_future_steps, 3)
            - "ego_future_rot": torch.Tensor of shape (1, 1, num_future_steps, 3, 3)
            - "relative_timestamps": torch.Tensor of shape (N_cameras, num_frames)
            - "absolute_timestamps": torch.Tensor of shape (N_cameras, num_frames)
            - "t0_us": int - the t0 timestamp used
            - "clip_id": str - the clip ID
            - "split": str - the dataset split ("train", "val", or "test")

    Examples:
        >>> # Load all clips
        >>> dataset = LocalPhysicalAIAVDataset("/path/to/dataset")

        >>> # Load only training split
        >>> train_dataset = LocalPhysicalAIAVDataset("/path/to/dataset", split="train")

        >>> # Load specific clips
        >>> dataset = LocalPhysicalAIAVDataset("/path/to/dataset", clip_ids=["clip1", "clip2"])
    """

    def __init__(
        self,
        dataset_root: str | pathlib.Path,
        clip_ids: list[str] | None = None,
        split: str | None = None,
        cameras: list[str] | None = None,
        t0_us: int = 0,
        dt_us: int = 50_000,
        num_frames: int = 4,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
    ) -> None:
        if clip_ids is not None and split is not None:
            raise ValueError("Cannot specify both clip_ids and split. Choose one.")

        self.dataset_root = pathlib.Path(dataset_root)
        self.cameras = cameras or ["camera_front_wide_120fov"]
        self.t0_us = t0_us
        self.dt_us = dt_us
        self.num_frames = num_frames
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps

        # Load metadata files
        self._load_metadata()

        # Determine which clips to use
        if clip_ids is not None:
            self.clip_ids = [cid for cid in clip_ids if cid in self.clip_index.index]
            if len(self.clip_ids) != len(clip_ids):
                missing = set(clip_ids) - set(self.clip_ids)
                logger.warning(f"Some clip IDs not found in dataset: {missing}")
        elif split is not None:
            if "split" not in self.clip_index.columns:
                raise ValueError("clip_index does not contain 'split' column")
            valid_splits = ["train", "val", "test"]
            if split not in valid_splits:
                raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
            self.clip_ids = self.clip_index[self.clip_index["split"] == split].index.tolist()
            logger.info(f"Filtered by split='{split}': {len(self.clip_ids)} clips")
        else:
            self.clip_ids = self.clip_index.index.tolist()

        logger.info(f"Initialized LocalPhysicalAIAVDataset with {len(self.clip_ids)} clips")

    def _load_metadata(self) -> None:
        """Load dataset metadata files."""
        # Load features.csv
        features_path = self.dataset_root / "features.csv"
        features_df = pd.read_csv(features_path, index_col="feature")
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = _Features(features_df)

        # Load clip index
        clip_index_path = self.dataset_root / "clip_index.parquet"
        self.clip_index = pd.read_parquet(clip_index_path)

        # Load sensor presence
        sensor_presence_path = self.dataset_root / "metadata" / "sensor_presence.parquet"
        self.sensor_presence = pd.read_parquet(sensor_presence_path)

        # Load data collection metadata
        data_collection_path = self.dataset_root / "metadata" / "data_collection.parquet"
        self.data_collection = pd.read_parquet(data_collection_path)

        # Build chunk sensor presence table
        self.chunk_sensor_presence = (
            pd.concat(
                [self.clip_index[["chunk"]], self.sensor_presence.select_dtypes(include=bool)],
                axis=1,
            )
            .groupby("chunk")
            .any()
        )

    def __len__(self) -> int:
        """Return the number of clips in the dataset."""
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single clip's data.

        Args:
            idx: Index of the clip to retrieve.

        Returns:
            Dictionary containing image frames and egomotion data.
        """
        clip_id = self.clip_ids[idx]
        return self.load_clip(clip_id)

    def get_clip_chunk(self, clip_id: str) -> int:
        """Returns the chunk index for a clip_id."""
        return self.clip_index.at[clip_id, "chunk"]

    def _get_chunk_feature_path(self, chunk_id: int, feature: str) -> pathlib.Path:
        """Get the local file path for a chunk feature."""
        relative_path = self.features.get_chunk_feature_filename(chunk_id, feature)
        return self.dataset_root / relative_path

    def _load_camera_data(
        self, clip_id: str, camera: str, timestamps: np.ndarray
    ) -> np.ndarray:
        """Load camera frames for given timestamps.

        Args:
            clip_id: The clip ID.
            camera: Camera name (e.g., "camera_front_wide_120fov").
            timestamps: Array of timestamps in microseconds.

        Returns:
            Array of shape (num_frames, H, W, C) containing RGB images.
        """
        chunk_id = self.get_clip_chunk(clip_id)
        chunk_path = self._get_chunk_feature_path(chunk_id, camera)

        if not chunk_path.exists():
            raise FileNotFoundError(f"Camera chunk file not found: {chunk_path}")

        clip_files_in_zip = self.features.get_clip_files_in_zip(clip_id, camera)

        with zipfile.ZipFile(chunk_path, "r") as zf:
            # Load video data
            video_data = io.BytesIO(zf.read(clip_files_in_zip["video"]))

            # Load timestamps
            frame_timestamps_df = pd.read_parquet(
                io.BytesIO(zf.read(clip_files_in_zip["frame_timestamps"]))
            )
            frame_timestamps = frame_timestamps_df["timestamp"].to_numpy(copy=True)

            # Create video reader and decode frames
            reader = video.SeekVideoReader(
                video_data=video_data,
                timestamps=frame_timestamps,
            )

            try:
                images, _ = reader.decode_images_from_timestamps(timestamps)
            finally:
                reader.close()

        return images

    def _load_egomotion_data(
        self, clip_id: str, timestamps: np.ndarray
    ) -> egomotion.EgomotionState:
        """Load and interpolate egomotion data for given timestamps.

        Args:
            clip_id: The clip ID.
            timestamps: Array of timestamps in microseconds.

        Returns:
            Interpolated EgomotionState at the requested timestamps.
        """
        chunk_id = self.get_clip_chunk(clip_id)
        chunk_path = self._get_chunk_feature_path(chunk_id, "egomotion")

        if not chunk_path.exists():
            raise FileNotFoundError(f"Egomotion chunk file not found: {chunk_path}")

        clip_files_in_zip = self.features.get_clip_files_in_zip(clip_id, "egomotion")

        with zipfile.ZipFile(chunk_path, "r") as zf:
            egomotion_df = pd.read_parquet(
                io.BytesIO(zf.read(clip_files_in_zip["egomotion"]))
            )

        # Create egomotion state and interpolator
        ego_state = egomotion.EgomotionState.from_egomotion_df(egomotion_df)
        interpolator = ego_state.create_interpolator(egomotion_df["timestamp"].to_numpy(copy=True))

        # Interpolate at requested timestamps
        return interpolator(timestamps)

    def load_clip(self, clip_id: str) -> dict[str, Any]:
        """Load all data for a single clip.

        Args:
            clip_id: The clip ID to load.

        Returns:
            Dictionary with image frames and egomotion data.
        """
        # Generate timestamps for image frames
        image_timestamps = np.array(
            [self.t0_us + i * self.dt_us for i in range(self.num_frames)],
            dtype=np.int64,
        )

        # Generate timestamps for history (before t0)
        history_timestamps = np.array(
            [self.t0_us - (self.num_history_steps - i) * self.dt_us for i in range(self.num_history_steps)],
            dtype=np.int64,
        )

        # Generate timestamps for future (after t0, including t0)
        future_timestamps = np.array(
            [self.t0_us + i * self.dt_us for i in range(self.num_future_steps)],
            dtype=np.int64,
        )

        # Load images from all cameras
        all_images = []
        for camera in self.cameras:
            images = self._load_camera_data(clip_id, camera, image_timestamps)
            all_images.append(images)

        # Stack images: (num_cameras, num_frames, H, W, C)
        image_frames = np.stack(all_images, axis=0)

        # Convert to torch.Tensor and permute to (N_cameras, num_frames, 3, H, W)
        # Input is (N_cameras, num_frames, H, W, C) where C=3 (RGB)
        image_frames = torch.from_numpy(image_frames).permute(0, 1, 4, 2, 3)

        # Create camera indices using the predefined mapping
        camera_indices = torch.tensor(
            [CAMERA_NAME_TO_INDEX.get(cam, -1) for cam in self.cameras],
            dtype=torch.long
        )

        # Load egomotion data for history
        ego_history_state = self._load_egomotion_data(clip_id, history_timestamps)
        ego_history_xyz = ego_history_state.pose.translation  # Shape: (num_history_steps, 3)
        ego_history_rot = ego_history_state.pose.rotation.as_matrix()  # Shape: (num_history_steps, 3, 3)

        # Load egomotion data for future
        ego_future_state = self._load_egomotion_data(clip_id, future_timestamps)
        ego_future_xyz = ego_future_state.pose.translation  # Shape: (num_future_steps, 3)
        ego_future_rot = ego_future_state.pose.rotation.as_matrix()  # Shape: (num_future_steps, 3, 3)

        # Reshape to (1, 1, num_steps, 3) for xyz
        ego_history_xyz_t = torch.from_numpy(ego_history_xyz).float().unsqueeze(0).unsqueeze(0)
        ego_future_xyz_t = torch.from_numpy(ego_future_xyz).float().unsqueeze(0).unsqueeze(0)

        # Reshape to (1, 1, num_steps, 3, 3) for rotation matrices
        ego_history_rot_t = torch.from_numpy(ego_history_rot).float().unsqueeze(0).unsqueeze(0)
        ego_future_rot_t = torch.from_numpy(ego_future_rot).float().unsqueeze(0).unsqueeze(0)

        # Create timestamps tensors for images
        absolute_timestamps = torch.from_numpy(image_timestamps).float().unsqueeze(0).expand(len(self.cameras), -1)
        relative_timestamps = absolute_timestamps - self.t0_us

        # Get split info if available
        split_info = self.clip_index.at[clip_id, "split"] if "split" in self.clip_index.columns else None

        return {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "ego_history_xyz": ego_history_xyz_t,
            "ego_history_rot": ego_history_rot_t,
            "ego_future_xyz": ego_future_xyz_t,
            "ego_future_rot": ego_future_rot_t,
            "relative_timestamps": relative_timestamps,
            "absolute_timestamps": absolute_timestamps,
            "t0_us": self.t0_us,
            "clip_id": clip_id,
            "split": split_info,
        }

    def get_clip_info(self, clip_id: str) -> pd.Series:
        """Get metadata information for a clip."""
        return self.clip_index.loc[clip_id]

    def get_sensor_presence(self, clip_id: str) -> pd.Series:
        """Get sensor presence information for a clip."""
        return self.sensor_presence.loc[clip_id]

    def load_calibration_data(
        self, clip_id: str
    ) -> tuple[CameraIntrinsics, SensorExtrinsics, VehicleDimensions]:
        """Load calibration data for a specific clip.

        Calibration data is stored as direct parquet files (not zipped) per chunk.
        Camera intrinsics and sensor extrinsics are indexed by (clip_id, sensor_name).
        Vehicle dimensions are indexed by clip_id.

        Args:
            clip_id: The clip ID to load calibration data for.

        Returns:
            Tuple of (camera_intrinsics, sensor_extrinsics, vehicle_dimensions).

        Example:
            >>> dataset = LocalPhysicalAIAVDataset("/path/to/data", clip_ids=["clip_id"])
            >>> camera_intrinsics, sensor_extrinsics, vehicle_dimensions = dataset.load_calibration_data("clip_id")
        """
        chunk_id = self.get_clip_chunk(clip_id)

        # Load camera intrinsics (direct parquet file, not zipped)
        # Filter by clip_id since intrinsics has MultiIndex (clip_id, camera_name)
        intrinsics_path = self._get_chunk_feature_path(chunk_id, "camera_intrinsics")
        intrinsics_df = pd.read_parquet(intrinsics_path)
        clip_intrinsics_df = intrinsics_df.loc[clip_id]  # Get only this clip's cameras
        camera_intrinsics = CameraIntrinsics.from_intrinsics_df(clip_intrinsics_df)

        # Load sensor extrinsics (direct parquet file, not zipped)
        # Filter by clip_id since extrinsics has MultiIndex (clip_id, sensor_name)
        extrinsics_path = self._get_chunk_feature_path(chunk_id, "sensor_extrinsics")
        extrinsics_df = pd.read_parquet(extrinsics_path)
        clip_extrinsics_df = extrinsics_df.loc[clip_id]  # Get only this clip's sensors
        sensor_extrinsics = SensorExtrinsics.from_extrinsics_df(clip_extrinsics_df)

        # Load vehicle dimensions (direct parquet file, not zipped)
        # Vehicle dimensions are indexed by clip_id, so look up the specific clip
        dimensions_path = self._get_chunk_feature_path(chunk_id, "vehicle_dimensions")
        dimensions_df = pd.read_parquet(dimensions_path)
        clip_dimensions = dimensions_df.loc[clip_id]
        vehicle_dimensions = VehicleDimensions(
            length=clip_dimensions["length"],
            width=clip_dimensions["width"],
            height=clip_dimensions["height"],
            rear_axle_to_bbox_center=clip_dimensions["rear_axle_to_bbox_center"],
            wheelbase=clip_dimensions["wheelbase"],
            track_width=clip_dimensions["track_width"],
        )

        return camera_intrinsics, sensor_extrinsics, vehicle_dimensions


class _Features:
    """Helper class for representing dataset features (mirrors Features in dataset.py)."""

    def __init__(self, features_df: pd.DataFrame) -> None:
        self.features_df = features_df

    def get_chunk_feature_filename(self, chunk_id: int, feature: str) -> str:
        """Returns the chunk feature filename within the dataset."""
        return self.features_df.at[feature, "chunk_path"].format(chunk_id=chunk_id)

    def get_clip_files_in_zip(self, clip_id: str, feature: str) -> dict[str, str]:
        """Returns the files within a chunk feature zip corresponding to clip_id."""
        templates = self.features_df.at[feature, "clip_files_in_zip"]
        if not isinstance(templates, dict):
            raise ValueError(f"{feature=} is not chunked as zip files.")
        return {k: v.format(clip_id=clip_id) for k, v in templates.items()}


def load_physical_aiavdataset(
    dataset_root: str | pathlib.Path,
    clip_id: str,
    t0_us: int = 0,
    dt_us: int = 50_000,
    num_frames: int = 4,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    cameras: list[str] | None = None,
) -> dict[str, Any]:
    """Load a single clip from local dataset (convenience function).

    This function provides a simple interface for loading a single clip,
    matching the API of the original HuggingFace-based loader.

    Args:
        dataset_root: Path to the dataset root directory.
        clip_id: The clip ID to load.
        t0_us: Initial timestamp offset in microseconds. Defaults to 0.
        dt_us: Time step between frames in microseconds. Defaults to 50_000 (20Hz).
        num_frames: Number of image frames to load. Defaults to 4.
        num_history_steps: Number of history egomotion steps. Defaults to 16.
        num_future_steps: Number of future egomotion steps. Defaults to 64.
        cameras: List of camera names to load. Defaults to ["camera_front_wide_120fov"].

    Returns:
        Dictionary containing:
            - "image_frames": torch.Tensor of shape (N_cameras, num_frames, 3, H, W)
            - "camera_indices": torch.Tensor of shape (N_cameras,)
            - "ego_history_xyz": torch.Tensor of shape (1, 1, num_history_steps, 3)
            - "ego_history_rot": torch.Tensor of shape (1, 1, num_history_steps, 3, 3)
            - "ego_future_xyz": torch.Tensor of shape (1, 1, num_future_steps, 3)
            - "ego_future_rot": torch.Tensor of shape (1, 1, num_future_steps, 3, 3)
            - "relative_timestamps": torch.Tensor of shape (N_cameras, num_frames)
            - "absolute_timestamps": torch.Tensor of shape (N_cameras, num_frames)
            - "t0_us": int - the t0 timestamp used
            - "clip_id": str
            - "split": str or None - the dataset split

    Example:
        >>> data = load_physical_aiavdataset(
        ...     "/path/to/dataset",
        ...     "030c760c-ae38-49aa-9ad8-f5650a545d26",
        ...     t0_us=5_100_000
        ... )
        >>> print(data["image_frames"].shape)
        (4, 4, 3, 1080, 1920)
        >>> print(data["ego_history_xyz"].shape)
        (1, 1, 16, 3)
    """
    dataset = LocalPhysicalAIAVDataset(
        dataset_root=dataset_root,
        clip_ids=[clip_id],
        cameras=cameras,
        t0_us=t0_us,
        dt_us=dt_us,
        num_frames=num_frames,
        num_history_steps=num_history_steps,
        num_future_steps=num_future_steps,
    )
    return dataset[0]


def get_dataloader(
    dataset_root: str | pathlib.Path,
    clip_ids: list[str] | None = None,
    split: str | None = None,
    cameras: list[str] | None = None,
    t0_us: int = 0,
    dt_us: int = 50_000,
    num_frames: int = 4,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the PhysicalAI-AV dataset.

    This is a convenience function to create a DataLoader with the specified
    dataset configuration.

    Args:
        dataset_root: Path to the dataset root directory.
        clip_ids: Optional list of clip IDs to load.
        split: Optional dataset split to filter by ("train", "val", or "test").
        cameras: List of camera names to load.
        t0_us: Initial timestamp offset in microseconds.
        dt_us: Time step between frames in microseconds.
        num_frames: Number of image frames to load per sample.
        num_history_steps: Number of history egomotion steps.
        num_future_steps: Number of future egomotion steps.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes (must be 0 due to video reader limitations).
        **kwargs: Additional arguments passed to DataLoader.

    Returns:
        A DataLoader instance.

    Example:
        >>> dataloader = get_dataloader(
        ...     "/path/to/dataset",
        ...     split="train",
        ...     batch_size=4,
        ...     cameras=["camera_front_wide_120fov"],
        ... )
        >>> for batch in dataloader:
        ...     print(batch["image_frames"].shape)
        ...     break
    """
    from torch.utils.data import DataLoader

    dataset = LocalPhysicalAIAVDataset(
        dataset_root=dataset_root,
        clip_ids=clip_ids,
        split=split,
        cameras=cameras,
        t0_us=t0_us,
        dt_us=dt_us,
        num_frames=num_frames,
        num_history_steps=num_history_steps,
        num_future_steps=num_future_steps,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )

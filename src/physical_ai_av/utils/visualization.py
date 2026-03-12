# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Visualization utilities for PhysicalAI-Autonomous-Vehicles dataset.

This module provides functions for visualizing samples from the dataset,
including camera images, egomotion trajectories, and vehicle bounding boxes.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as spt

from physical_ai_av.calibration import CameraIntrinsics, SensorExtrinsics, VehicleDimensions
from physical_ai_av.utils import camera_models as cm
from physical_ai_av.utils import tf

if TYPE_CHECKING:
    import torch


def visualize_sample(
    data: dict,
    camera_intrinsics: CameraIntrinsics | None = None,
    sensor_extrinsics: SensorExtrinsics | None = None,
    vehicle_dimensions: VehicleDimensions | None = None,
    camera_names: list[str] | None = None,
    figsize: tuple[int, int] = (20, 12),
    show_egomotion: bool = True,
    show_bbox: bool = True,
    lookahead_time_s: float = 5.0,
    num_lookahead_points: int = 50,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Visualize a single sample from the dataset.

    Creates a visualization showing camera images with overlaid egomotion
    trajectories and vehicle bounding boxes projected into camera frames.

    Args:
        data: Dictionary containing sample data from LocalPhysicalAIAVDataset.
            Expected keys:
            - "image_frames": torch.Tensor of shape (N_cameras, num_frames, 3, H, W)
            - "camera_indices": torch.Tensor of shape (N_cameras,)
            - "ego_history_xyz": torch.Tensor of shape (1, 1, num_history_steps, 3)
            - "ego_history_rot": torch.Tensor of shape (1, 1, num_history_steps, 3, 3)
            - "ego_future_xyz": torch.Tensor of shape (1, 1, num_future_steps, 3)
            - "ego_future_rot": torch.Tensor of shape (1, 1, num_future_steps, 3, 3)
            - "relative_timestamps": torch.Tensor of shape (N_cameras, num_frames)
            - "absolute_timestamps": torch.Tensor of shape (N_cameras, num_frames)
            - "t0_us": int - the t0 timestamp
            - "clip_id": str
            - "split": str
        camera_intrinsics: CameraIntrinsics object containing camera models.
        sensor_extrinsics: SensorExtrinsics object containing sensor poses.
        vehicle_dimensions: VehicleDimensions object containing vehicle dimensions.
        camera_names: List of camera names corresponding to camera_indices.
            If None, will try to infer from camera_indices.
        figsize: Figure size for the plot.
        show_egomotion: Whether to show egomotion trajectory overlay.
        show_bbox: Whether to show vehicle bounding box overlay.
        lookahead_time_s: Lookahead time in seconds for trajectory visualization.
        num_lookahead_points: Number of points to sample for trajectory visualization.

    Returns:
        Tuple of (figure, list of axes) for the created plots.

    Example:
        >>> from physical_ai_av import LocalPhysicalAIAVDataset
        >>> from physical_ai_av.utils.visualization import visualize_sample
        >>> dataset = LocalPhysicalAIAVDataset("/path/to/data", clip_ids=["clip_id"])
        >>> data = dataset[0]
        >>> fig, axes = visualize_sample(data)
        >>> plt.show()
    """
    # Extract data
    image_frames = data["image_frames"]  # (N_cameras, num_frames, 3, H, W)
    camera_indices = data["camera_indices"]  # (N_cameras,)
    ego_future_xyz = data["ego_future_xyz"]  # (1, 1, num_future_steps, 3)
    ego_future_rot = data["ego_future_rot"]  # (1, 1, num_future_steps, 3, 3)
    t0_us = data["t0_us"]
    clip_id = data["clip_id"]

    num_cameras = image_frames.shape[0]

    # Default camera names mapping
    if camera_names is None:
        index_to_camera = {
            0: "camera_cross_left_120fov",
            1: "camera_front_wide_120fov",
            2: "camera_cross_right_120fov",
            3: "camera_rear_left_70fov",
            4: "camera_rear_tele_30fov",
            5: "camera_rear_right_70fov",
            6: "camera_front_tele_30fov",
        }
        camera_names = [index_to_camera.get(int(idx), f"camera_{idx}") for idx in camera_indices]

    # Create figure with subplots
    ncols = min(2, num_cameras)
    nrows = (num_cameras + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_cameras == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else list(axes)

    # Get the last frame from each camera for visualization
    # image_frames shape: (N_cameras, num_frames, 3, H, W)
    last_frames = image_frames[:, -1, ...]  # (N_cameras, 3, H, W)

    # Convert from torch tensor to numpy if needed
    if hasattr(last_frames, "numpy"):
        last_frames = last_frames.numpy()

    # Convert from (C, H, W) to (H, W, C) for display
    last_frames = np.transpose(last_frames, (0, 2, 3, 1))

    # Normalize to [0, 1] if needed
    if last_frames.max() > 1.0:
        last_frames = last_frames / 255.0

    # Create egomotion interpolator if calibration data is available
    tf_tree = None
    if show_egomotion or show_bbox:
        if sensor_extrinsics is not None:
            tf_tree = _create_transform_tree(
                ego_future_xyz, ego_future_rot, t0_us, sensor_extrinsics
            )

    # Plot each camera
    for i, (ax, camera_name) in enumerate(zip(axes, camera_names)):
        ax.imshow(last_frames[i])
        ax.set_title(f"{camera_name}\nClip: {clip_id}")
        ax.axis("off")

        # Skip overlays if calibration data is not available
        if tf_tree is None or camera_intrinsics is None:
            continue

        # Get camera model
        if camera_name not in camera_intrinsics.camera_models:
            continue
        camera_model = camera_intrinsics.camera_models[camera_name]

        # Get transform from anchor to camera frame at t0
        frame_time = t0_us
        try:
            anchor_to_camera_frame = tf_tree.lookup_transform(
                tf.FrameInfo(camera_name, frame_time), tf.FrameInfo("anchor")
            )
        except (ValueError, KeyError):
            continue

        # Show egomotion trajectory
        if show_egomotion:
            _plot_egomotion_trajectory(
                ax,
                ego_future_xyz,
                ego_future_rot,
                t0_us,
                anchor_to_camera_frame,
                camera_model,
                lookahead_time_s,
                num_lookahead_points,
            )

        # Show vehicle bounding box
        if show_bbox and vehicle_dimensions is not None:
            _plot_vehicle_bbox(
                ax,
                ego_future_xyz,
                ego_future_rot,
                t0_us,
                anchor_to_camera_frame,
                camera_model,
                vehicle_dimensions,
                lookahead_time_s,
                num_lookahead_points,
            )

    # Hide unused subplots
    for i in range(num_cameras, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig, axes


def _create_transform_tree(
    ego_future_xyz: "torch.Tensor | np.ndarray",
    ego_future_rot: "torch.Tensor | np.ndarray",
    t0_us: int,
    sensor_extrinsics: SensorExtrinsics,
) -> tf.TransformTree:
    """Create a transform tree with egomotion and sensor extrinsics.

    Args:
        ego_future_xyz: Future ego positions, shape (1, 1, num_future_steps, 3).
        ego_future_rot: Future ego rotations, shape (1, 1, num_future_steps, 3, 3).
        t0_us: Reference timestamp in microseconds.
        sensor_extrinsics: SensorExtrinsics containing sensor poses.

    Returns:
        TransformTree with anchor->ego and ego->sensor transforms.
    """
    # Convert to numpy if needed
    if hasattr(ego_future_xyz, "numpy"):
        ego_future_xyz = ego_future_xyz.numpy()
    if hasattr(ego_future_rot, "numpy"):
        ego_future_rot = ego_future_rot.numpy()

    # Squeeze batch dimensions
    ego_future_xyz = ego_future_xyz.squeeze()  # (num_future_steps, 3)
    ego_future_rot = ego_future_rot.squeeze()  # (num_future_steps, 3, 3)

    # Create timestamps for future steps (assuming 20Hz = 50ms = 50000us)
    dt_us = 50000
    num_future_steps = ego_future_xyz.shape[0]
    timestamps = np.array([t0_us + i * dt_us for i in range(num_future_steps)])

    # Create egomotion RigidTransform
    ego_pose = spt.RigidTransform.from_components(
        rotation=spt.Rotation.from_matrix(ego_future_rot),
        translation=ego_future_xyz,
    )

    # Create interpolator for egomotion
    from physical_ai_av.utils import interpolation

    ego_interpolator = interpolation.RigidTransformInterpolator(timestamps, ego_pose)

    # Create transform tree
    tf_tree = tf.TransformTree(root_frame_id="anchor")
    tf_tree.add_transform("anchor", "ego", ego_interpolator)

    # Add sensor extrinsics (static transforms)
    for sensor_id, sensor_pose in sensor_extrinsics.sensor_poses.items():
        tf_tree.add_transform("ego", sensor_id, sensor_pose)

    return tf_tree


def _plot_egomotion_trajectory(
    ax: plt.Axes,
    ego_future_xyz: "torch.Tensor | np.ndarray",
    ego_future_rot: "torch.Tensor | np.ndarray",
    t0_us: int,
    anchor_to_camera_frame: tf.FrameTransform,
    camera_model: cm.CameraModel,
    lookahead_time_s: float,
    num_lookahead_points: int,
) -> None:
    """Plot egomotion trajectory on the camera image.

    Args:
        ax: Matplotlib axes to plot on.
        ego_future_xyz: Future ego positions.
        ego_future_rot: Future ego rotations.
        t0_us: Reference timestamp in microseconds.
        anchor_to_camera_frame: Transform from anchor to camera frame.
        camera_model: Camera model for projection.
        lookahead_time_s: Lookahead time in seconds.
        num_lookahead_points: Number of points to sample.
    """
    # Convert to numpy if needed
    if hasattr(ego_future_xyz, "numpy"):
        ego_future_xyz = ego_future_xyz.numpy()
    if hasattr(ego_future_rot, "numpy"):
        ego_future_rot = ego_future_rot.numpy()

    # Squeeze batch dimensions
    ego_future_xyz = ego_future_xyz.squeeze()  # (num_future_steps, 3)
    ego_future_rot = ego_future_rot.squeeze()  # (num_future_steps, 3, 3)

    # Generate query timestamps
    dt_us = 50000
    num_future_steps = ego_future_xyz.shape[0]
    frame_time = t0_us
    max_time_us = min(int(lookahead_time_s * 1e6), (num_future_steps - 1) * dt_us)
    query_times = np.linspace(frame_time, frame_time + max_time_us, num_lookahead_points)

    # Get egomotion poses at query times
    query_indices = np.clip(
        ((query_times - t0_us) / dt_us).astype(int), 0, num_future_steps - 1
    )
    ego_positions = ego_future_xyz[query_indices]  # (num_points, 3)

    # Transform to camera frame
    ego_in_camera_frame = anchor_to_camera_frame.tf_target_source.apply(ego_positions)

    # Project to image plane
    egomotion_pixels = camera_model.ray2pixel(ego_in_camera_frame)

    # Filter out-of-bounds points
    out_of_bounds = camera_model.is_out_of_bounds(egomotion_pixels)
    egomotion_pixels = np.where(out_of_bounds[:, None], np.nan, egomotion_pixels)

    # Plot trajectory with color indicating lookahead time
    valid_mask = ~np.isnan(egomotion_pixels).any(axis=1)
    if valid_mask.any():
        lookahead_times = (query_times - frame_time) / 1e6
        scatter = ax.scatter(
            egomotion_pixels[valid_mask, 0],
            egomotion_pixels[valid_mask, 1],
            c=lookahead_times[valid_mask],
            s=20,
            cmap="rainbow",
            alpha=0.8,
        )
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Lookahead Time (s)", fraction=0.046, pad=0.04)


def _plot_vehicle_bbox(
    ax: plt.Axes,
    ego_future_xyz: "torch.Tensor | np.ndarray",
    ego_future_rot: "torch.Tensor | np.ndarray",
    t0_us: int,
    anchor_to_camera_frame: tf.FrameTransform,
    camera_model: cm.CameraModel,
    vehicle_dimensions: VehicleDimensions,
    lookahead_time_s: float,
    num_lookahead_points: int,
) -> None:
    """Plot vehicle bounding box trajectory on the camera image.

    Args:
        ax: Matplotlib axes to plot on.
        ego_future_xyz: Future ego positions.
        ego_future_rot: Future ego rotations.
        t0_us: Reference timestamp in microseconds.
        anchor_to_camera_frame: Transform from anchor to camera frame.
        camera_model: Camera model for projection.
        vehicle_dimensions: VehicleDimensions object.
        lookahead_time_s: Lookahead time in seconds.
        num_lookahead_points: Number of points to sample.
    """
    # Convert to numpy if needed
    if hasattr(ego_future_xyz, "numpy"):
        ego_future_xyz = ego_future_xyz.numpy()
    if hasattr(ego_future_rot, "numpy"):
        ego_future_rot = ego_future_rot.numpy()

    # Squeeze batch dimensions
    ego_future_xyz = ego_future_xyz.squeeze()  # (num_future_steps, 3)
    ego_future_rot = ego_future_rot.squeeze()  # (num_future_steps, 3, 3)

    # Generate query timestamps
    dt_us = 50000
    num_future_steps = ego_future_xyz.shape[0]
    frame_time = t0_us
    max_time_us = min(int(lookahead_time_s * 1e6), (num_future_steps - 1) * dt_us)
    query_times = np.linspace(frame_time, frame_time + max_time_us, num_lookahead_points)

    # Get egomotion indices at query times
    query_indices = np.clip(
        ((query_times - t0_us) / dt_us).astype(int), 0, num_future_steps - 1
    )

    # Create bounding box corners in ego frame
    # Rectangle in XY plane at Z=0
    bbox_corners_xyz_ego = np.array(
        [
            [vehicle_dimensions.length / 2, vehicle_dimensions.width / 2, 0],
            [vehicle_dimensions.length / 2, -vehicle_dimensions.width / 2, 0],
            [-vehicle_dimensions.length / 2, -vehicle_dimensions.width / 2, 0],
            [-vehicle_dimensions.length / 2, vehicle_dimensions.width / 2, 0],
            [vehicle_dimensions.length / 2, vehicle_dimensions.width / 2, 0],  # Close the loop
        ]
    )
    # Adjust for rear axle offset
    bbox_corners_xyz_ego[..., 0] -= vehicle_dimensions.rear_axle_to_bbox_center

    # Transform bbox corners to anchor frame for each timestep
    cmap = plt.cm.rainbow
    norm = plt.cm.colors.Normalize(vmin=0, vmax=lookahead_time_s)

    for i, idx in enumerate(query_indices):
        # Get ego pose at this timestep
        ego_position = ego_future_xyz[idx]
        ego_rotation = ego_future_rot[idx]

        # Transform bbox corners to anchor frame
        bbox_corners_xyz_anchor = (
            ego_rotation @ bbox_corners_xyz_ego.T
        ).T + ego_position

        # Transform to camera frame
        bbox_corners_xyz_camera = anchor_to_camera_frame.tf_target_source.apply(
            bbox_corners_xyz_anchor
        )

        # Project to image plane
        bbox_corners_uv = camera_model.ray2pixel(bbox_corners_xyz_camera)

        # Check if any corner is out of bounds
        out_of_bounds = camera_model.is_out_of_bounds(bbox_corners_uv).any()
        if out_of_bounds:
            continue

        # Plot bounding box
        lookahead_time = (query_times[i] - frame_time) / 1e6
        color = cmap(norm(lookahead_time))
        ax.plot(
            bbox_corners_uv[:, 0],
            bbox_corners_uv[:, 1],
            color=color,
            linewidth=2,
            alpha=0.6,
        )


def visualize_egomotion_trajectory_3d(
    data: dict,
    figsize: tuple[int, int] = (10, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize egomotion trajectory in 3D.

    Args:
        data: Dictionary containing sample data.
        figsize: Figure size for the plot.

    Returns:
        Tuple of (figure, axes) for the created plot.
    """
    ego_history_xyz = data["ego_history_xyz"]
    ego_future_xyz = data["ego_future_xyz"]
    t0_us = data["t0_us"]

    # Convert to numpy if needed
    if hasattr(ego_history_xyz, "numpy"):
        ego_history_xyz = ego_history_xyz.numpy()
    if hasattr(ego_future_xyz, "numpy"):
        ego_future_xyz = ego_future_xyz.numpy()

    # Squeeze batch dimensions
    ego_history_xyz = ego_history_xyz.squeeze()  # (num_history_steps, 3)
    ego_future_xyz = ego_future_xyz.squeeze()  # (num_future_steps, 3)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot history trajectory
    ax.plot(
        ego_history_xyz[:, 0],
        ego_history_xyz[:, 1],
        ego_history_xyz[:, 2],
        "b-o",
        label="History",
        markersize=4,
    )

    # Plot future trajectory
    ax.plot(
        ego_future_xyz[:, 0],
        ego_future_xyz[:, 1],
        ego_future_xyz[:, 2],
        "r-o",
        label="Future",
        markersize=4,
    )

    # Mark t0 position
    ax.scatter(
        [ego_future_xyz[0, 0]],
        [ego_future_xyz[0, 1]],
        [ego_future_xyz[0, 2]],
        c="g",
        s=100,
        marker="*",
        label="t0",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Egomotion Trajectory\nClip: {data['clip_id']}, t0: {t0_us}us")
    ax.legend()
    ax.grid(True)

    # Set equal aspect ratio
    max_range = np.array(
        [
            ego_history_xyz[:, 0].max() - ego_history_xyz[:, 0].min(),
            ego_history_xyz[:, 1].max() - ego_history_xyz[:, 1].min(),
            ego_history_xyz[:, 2].max() - ego_history_xyz[:, 2].min(),
            ego_future_xyz[:, 0].max() - ego_future_xyz[:, 0].min(),
            ego_future_xyz[:, 1].max() - ego_future_xyz[:, 1].min(),
            ego_future_xyz[:, 2].max() - ego_future_xyz[:, 2].min(),
        ]
    ).max() / 2.0

    mid_x = (ego_history_xyz[:, 0].mean() + ego_future_xyz[:, 0].mean()) / 2
    mid_y = (ego_history_xyz[:, 1].mean() + ego_future_xyz[:, 1].mean()) / 2
    mid_z = (ego_history_xyz[:, 2].mean() + ego_future_xyz[:, 2].mean()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return fig, ax


def visualize_trajectory_front_view(
    data: dict,
    camera_intrinsics: CameraIntrinsics,
    sensor_extrinsics: SensorExtrinsics,
    vehicle_dimensions: VehicleDimensions,
    camera_name: str = "camera_front_wide_120fov",
    lookahead_time_s: float = 5.0,
    num_lookahead_points: int = 50,
    figsize: tuple[int, int] = (12, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize trajectory in front view (camera perspective) with bounding boxes.

    This function implements the same visualization logic as the data_visualization.ipynb
    notebook, projecting the vehicle's future trajectory and bounding boxes into a
    front-facing camera image.

    Args:
        data: Dictionary containing sample data from LocalPhysicalAIAVDataset.
        camera_intrinsics: CameraIntrinsics object containing camera models.
        sensor_extrinsics: SensorExtrinsics object containing sensor poses.
        vehicle_dimensions: VehicleDimensions object containing vehicle dimensions.
        camera_name: Name of the front camera to use for visualization.
        lookahead_time_s: Lookahead time in seconds for trajectory visualization.
        num_lookahead_points: Number of points to sample for trajectory visualization.
        figsize: Figure size for the plot.

    Returns:
        Tuple of (figure, axes) for the created plot.

    Example:
        >>> from physical_ai_av import LocalPhysicalAIAVDataset
        >>> from physical_ai_av.utils.visualization import visualize_trajectory_front_view
        >>> dataset = LocalPhysicalAIAVDataset("/path/to/data", clip_ids=["clip_id"])
        >>> data = dataset[0]
        >>> fig, ax = visualize_trajectory_front_view(
        ...     data, camera_intrinsics, sensor_extrinsics, vehicle_dimensions
        ... )
        >>> plt.show()
    """
    # Extract data
    image_frames = data["image_frames"]  # (N_cameras, num_frames, 3, H, W)
    camera_indices = data["camera_indices"]  # (N_cameras,)
    ego_future_xyz = data["ego_future_xyz"]  # (1, 1, num_future_steps, 3)
    ego_future_rot = data["ego_future_rot"]  # (1, 1, num_future_steps, 3, 3)
    t0_us = data["t0_us"]
    clip_id = data["clip_id"]

    # Find the index of the specified camera
    camera_names = []
    index_to_camera = {
        0: "camera_cross_left_120fov",
        1: "camera_front_wide_120fov",
        2: "camera_cross_right_120fov",
        3: "camera_rear_left_70fov",
        4: "camera_rear_tele_30fov",
        5: "camera_rear_right_70fov",
        6: "camera_front_tele_30fov",
    }
    for idx in camera_indices:
        camera_names.append(index_to_camera.get(int(idx), f"camera_{idx}"))

    if camera_name not in camera_names:
        raise ValueError(f"Camera {camera_name} not found in data. Available: {camera_names}")

    camera_idx = camera_names.index(camera_name)

    # Get the last frame from the specified camera
    last_frame = image_frames[camera_idx, -1, ...]  # (3, H, W)

    # Convert from torch tensor to numpy if needed
    if hasattr(last_frame, "numpy"):
        last_frame = last_frame.numpy()

    # Convert from (C, H, W) to (H, W, C) for display
    last_frame = np.transpose(last_frame, (1, 2, 0))

    # Normalize to [0, 1] if needed
    if last_frame.max() > 1.0:
        last_frame = last_frame / 255.0

    # Create transform tree
    tf_tree = _create_transform_tree(ego_future_xyz, ego_future_rot, t0_us, sensor_extrinsics)

    # Get camera model
    if camera_name not in camera_intrinsics.camera_models:
        raise ValueError(f"Camera {camera_name} not found in camera_intrinsics")
    camera_model = camera_intrinsics.camera_models[camera_name]

    # Get transform from anchor to camera frame at t0
    frame_time = t0_us
    anchor_to_camera_frame = tf_tree.lookup_transform(
        tf.FrameInfo(camera_name, frame_time), tf.FrameInfo("anchor")
    )

    # Convert to numpy and squeeze
    if hasattr(ego_future_xyz, "numpy"):
        ego_future_xyz = ego_future_xyz.numpy()
    if hasattr(ego_future_rot, "numpy"):
        ego_future_rot = ego_future_rot.numpy()
    ego_future_xyz = ego_future_xyz.squeeze()  # (num_future_steps, 3)
    ego_future_rot = ego_future_rot.squeeze()  # (num_future_steps, 3, 3)

    # Generate query timestamps
    dt_us = 50000
    num_future_steps = ego_future_xyz.shape[0]
    max_time_us = min(int(lookahead_time_s * 1e6), (num_future_steps - 1) * dt_us)
    query_times = np.linspace(t0_us, t0_us + max_time_us, num_lookahead_points)

    # Get egomotion indices at query times
    query_indices = np.clip(
        ((query_times - t0_us) / dt_us).astype(int), 0, num_future_steps - 1
    )

    # Create figure
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(last_frame)

    # Create colormap and normalizer for the lookahead time
    cmap = plt.cm.rainbow
    norm = plt.cm.colors.Normalize(vmin=0, vmax=lookahead_time_s)

    # Create bounding box corners in ego frame (same as notebook)
    bbox_corners_xyz_ego_frame = np.array(
        [
            [vehicle_dimensions.length / 2, vehicle_dimensions.width / 2, 0],
            [vehicle_dimensions.length / 2, -vehicle_dimensions.width / 2, 0],
            [-vehicle_dimensions.length / 2, -vehicle_dimensions.width / 2, 0],
            [-vehicle_dimensions.length / 2, vehicle_dimensions.width / 2, 0],
            [vehicle_dimensions.length / 2, vehicle_dimensions.width / 2, 0],  # Close the loop
        ]
    )
    # Adjust for rear axle offset
    bbox_corners_xyz_ego_frame[..., 0] -= vehicle_dimensions.rear_axle_to_bbox_center

    # Transform and project bounding boxes for each timestep
    for i, idx in enumerate(query_indices):
        # Get ego pose at this timestep
        ego_position = ego_future_xyz[idx]
        ego_rotation = ego_future_rot[idx]

        # Transform bbox corners to anchor frame
        # bbox_corners_xyz_anchor_frame = egomotion(query_times).pose.apply(bbox_corners_xyz_ego_frame[:, None, :])
        bbox_corners_xyz_anchor_frame = (
            ego_rotation @ bbox_corners_xyz_ego_frame.T
        ).T + ego_position

        # Transform to camera frame and reshape
        bbox_corners_xyz_camera_frame = anchor_to_camera_frame.tf_target_source.apply(
            bbox_corners_xyz_anchor_frame.reshape(-1, 3)
        ).reshape(5, -1, 3)

        # Project to image plane
        bbox_corners_uv_camera_image = camera_model.ray2pixel(bbox_corners_xyz_camera_frame)

        # Don't draw out of bounds bounding boxes
        out_of_bounds = camera_model.is_out_of_bounds(bbox_corners_uv_camera_image).any(axis=0)[
            None, :, None
        ]
        bbox_corners_uv_camera_image = np.where(
            out_of_bounds, np.nan, bbox_corners_uv_camera_image
        )

        # Plot bounding box with color scaled by lookahead time
        lookahead_time = (query_times[i] - t0_us) / 1e6
        color = cmap(norm(lookahead_time))
        ax.plot(
            bbox_corners_uv_camera_image[:, 0, 0],
            bbox_corners_uv_camera_image[:, 0, 1],
            color=color,
            linewidth=2,
        )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="Lookahead Time (s)", ax=ax)

    ax.set_title(f"Trajectory Front View\n{camera_name}\nClip: {clip_id}")
    ax.axis("off")
    plt.tight_layout()

    return fig, ax

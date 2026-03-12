#!/usr/bin/env python3
"""Usage example for LocalPhysicalAIAVDataset.

This script loads clip 25cd4769-5dcf-4b53-a351-bf2c5deb6124 from chunk 0
with 4 cameras using LocalPhysicalAIAVDataset.
"""

from src.physical_ai_av import LocalPhysicalAIAVDataset
from src.physical_ai_av.utils.visualization import (
    visualize_egomotion_trajectory_3d,
    visualize_sample,
    visualize_trajectory_front_view,
)

DATASET_ROOT = "/home/duzl/hlc/physical_ai_av/data"


def main():
    """Load clip 25cd4769-5dcf-4b53-a351-bf2c5deb6124 with LocalPhysicalAIAVDataset."""
    print("=" * 60)
    print("Loading clip 25cd4769-5dcf-4b53-a351-bf2c5deb6124")
    print("=" * 60)

    # Create dataset with specific clip and cameras
    dataset = LocalPhysicalAIAVDataset(
        DATASET_ROOT,
        clip_ids=["25cd4769-5dcf-4b53-a351-bf2c5deb6124"],
        cameras=[
            "camera_cross_left_120fov",
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
        ],
        t0_us=5_100_000,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Cameras: {dataset.cameras}")
    print()

    # Load the data
    data = dataset[0]

    # Load calibration data
    print("Loading calibration data...")
    camera_intrinsics, sensor_extrinsics, vehicle_dimensions = dataset.load_calibration_data(
        data["clip_id"]
    )
    print("Calibration data loaded.")
    print()

    def print_data_info(data):
      print("Data keys:", list(data.keys()))
      print()
      print(f"image_frames: {data['image_frames'].shape}")
      print(f"  -> (N_cameras, num_frames, 3, H, W)")
      print()
      print(f"camera_indices: {data['camera_indices'].shape}")
      print(f"  -> {data['camera_indices']}")
      print()
      print(f"ego_history_xyz: {data['ego_history_xyz'].shape}")
      print(f"  -> (1, 1, num_history_steps, 3)")
      print()
      print(f"ego_history_rot: {data['ego_history_rot'].shape}")
      print(f"  -> (1, 1, num_history_steps, 3, 3)")
      print()
      print(f"ego_future_xyz: {data['ego_future_xyz'].shape}")
      print(f"  -> (1, 1, num_future_steps, 3)")
      print()
      print(f"ego_future_rot: {data['ego_future_rot'].shape}")
      print(f"  -> (1, 1, num_future_steps, 3, 3)")
      print()
      print(f"relative_timestamps: {data['relative_timestamps'].shape}")
      print(f"  -> (N_cameras, num_frames)")
      print()
      print(f"absolute_timestamps: {data['absolute_timestamps'].shape}")
      print(f"  -> (N_cameras, num_frames)")
      print()
      print(f"t0_us: {data['t0_us']}")
      print(f"clip_id: {data['clip_id']}")
      print(f"split: {data['split']}")
      print()
      print("=" * 60)
      print("Successfully loaded!")
      print("=" * 60)

    print_data_info(data)

    # Visualize the sample
    print()
    print("=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # 1. Visualize camera images with egomotion and bounding box overlays
    fig, axes = visualize_sample(
        data,
        camera_intrinsics=camera_intrinsics,
        sensor_extrinsics=sensor_extrinsics,
        vehicle_dimensions=vehicle_dimensions,
        camera_names=dataset.cameras,
        show_egomotion=True,
        show_bbox=True,
        lookahead_time_s=5.0,
    )
    fig.savefig("visualization_camera_overlays.png", dpi=150, bbox_inches="tight")
    print("Saved: visualization_camera_overlays.png")

    # 2. Visualize 3D egomotion trajectory
    fig, ax = visualize_egomotion_trajectory_3d(data)
    fig.savefig("visualization_3d_trajectory.png", dpi=150, bbox_inches="tight")
    print("Saved: visualization_3d_trajectory.png")

    # 3. Visualize trajectory in front view (camera perspective) with bounding boxes
    fig, ax = visualize_trajectory_front_view(
        data,
        camera_intrinsics=camera_intrinsics,
        sensor_extrinsics=sensor_extrinsics,
        vehicle_dimensions=vehicle_dimensions,
        camera_name="camera_front_wide_120fov",
        lookahead_time_s=5.0,
    )
    fig.savefig("visualization_front_view.png", dpi=150, bbox_inches="tight")
    print("Saved: visualization_front_view.png")

    print()
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()


"""
============================================================
Loading clip 25cd4769-5dcf-4b53-a351-bf2c5deb6124
============================================================
Dataset size: 1
Cameras: ['camera_cross_left_120fov', 'camera_front_wide_120fov', 'camera_cross_right_120fov', 'camera_front_tele_30fov']

Data keys: ['image_frames', 'camera_indices', 'ego_history_xyz', 'ego_history_rot', 'ego_future_xyz', 'ego_future_rot', 'relative_timestamps', 'absolute_timestamps', 't0_us', 'clip_id', 'split']

image_frames: torch.Size([4, 4, 3, 1080, 1920])
  -> (N_cameras, num_frames, 3, H, W)

camera_indices: torch.Size([4])
  -> tensor([0, 1, 2, 6])

ego_history_xyz: torch.Size([1, 1, 16, 3])
  -> (1, 1, num_history_steps, 3)

ego_history_rot: torch.Size([1, 1, 16, 3, 3])
  -> (1, 1, num_history_steps, 3, 3)

ego_future_xyz: torch.Size([1, 1, 64, 3])
  -> (1, 1, num_future_steps, 3)

ego_future_rot: torch.Size([1, 1, 64, 3, 3])
  -> (1, 1, num_future_steps, 3, 3)

relative_timestamps: torch.Size([4, 4])
  -> (N_cameras, num_frames)

absolute_timestamps: torch.Size([4, 4])
  -> (N_cameras, num_frames)

t0_us: 5100000
clip_id: 25cd4769-5dcf-4b53-a351-bf2c5deb6124
split: train
"""
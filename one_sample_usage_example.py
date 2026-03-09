#!/usr/bin/env python3
"""Usage example for LocalPhysicalAIAVDataset.

This script loads clip 25cd4769-5dcf-4b53-a351-bf2c5deb6124 from chunk 0
with 4 cameras using LocalPhysicalAIAVDataset.
"""

from src.physical_ai_av import LocalPhysicalAIAVDataset

DATASET_ROOT = "/home/duzl/hlc/physical_ai_av_usage/data"


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
        num_frames=10,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Cameras: {dataset.cameras}")
    print()

    # Load the data
    data = dataset[0]

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


if __name__ == "__main__":
    main()

"""
============================================================
Loading clip 25cd4769-5dcf-4b53-a351-bf2c5deb6124
============================================================
Dataset size: 1
Cameras: ['camera_cross_left_120fov', 'camera_front_wide_120fov', 'camera_cross_right_120fov', 'camera_front_tele_30fov']

Data keys: ['image_frames', 'camera_indices', 'ego_history_xyz', 'ego_history_rot', 'ego_future_xyz', 'ego_future_rot', 'relative_timestamps', 'absolute_timestamps', 't0_us', 'clip_id', 'split']

image_frames: torch.Size([4, 10, 3, 1080, 1920])
  -> (N_cameras, num_frames, 3, H, W)

camera_indices: torch.Size([4])
  -> tensor([0, 1, 2, 3])

ego_history_xyz: torch.Size([1, 1, 10, 3])
  -> (1, 1, num_history_steps, 3)

ego_history_rot: torch.Size([1, 1, 10, 3, 3])
  -> (1, 1, num_history_steps, 3, 3)

ego_future_xyz: torch.Size([1, 1, 10, 3])
  -> (1, 1, num_future_steps, 3)

ego_future_rot: torch.Size([1, 1, 10, 3, 3])
  -> (1, 1, num_future_steps, 3, 3)

relative_timestamps: torch.Size([4, 10])
  -> (N_cameras, num_frames)

absolute_timestamps: torch.Size([4, 10])
  -> (N_cameras, num_frames)

t0_us: 5100000
clip_id: 25cd4769-5dcf-4b53-a351-bf2c5deb6124
split: train

============================================================
Successfully loaded!
============================================================
$ ^C

$  cd /home/duzl/hlc/physical_ai_av ; /usr/bin/env /home/duzl/.conda/envs/alpamayo1/bin/python /home/duzl/.qoder-server/extensions/ms-python.debugpy-2025.14.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 39881 -- /home/duzl/hlc/physical_ai_av/usage_example.py 
============================================================
Loading clip 25cd4769-5dcf-4b53-a351-bf2c5deb6124
============================================================
Dataset size: 1
Cameras: ['camera_cross_left_120fov', 'camera_front_wide_120fov', 'camera_cross_right_120fov', 'camera_front_tele_30fov']

Data keys: ['image_frames', 'camera_indices', 'ego_history_xyz', 'ego_history_rot', 'ego_future_xyz', 'ego_future_rot', 'relative_timestamps', 'absolute_timestamps', 't0_us', 'clip_id', 'split']

image_frames: torch.Size([4, 10, 3, 1080, 1920])
  -> (N_cameras, num_frames, 3, H, W)

camera_indices: torch.Size([4])
  -> tensor([0, 1, 2, 6])

ego_history_xyz: torch.Size([1, 1, 10, 3])
  -> (1, 1, num_history_steps, 3)

ego_history_rot: torch.Size([1, 1, 10, 3, 3])
  -> (1, 1, num_history_steps, 3, 3)

ego_future_xyz: torch.Size([1, 1, 10, 3])
  -> (1, 1, num_future_steps, 3)

ego_future_rot: torch.Size([1, 1, 10, 3, 3])
  -> (1, 1, num_future_steps, 3, 3)

relative_timestamps: torch.Size([4, 10])
  -> (N_cameras, num_frames)

absolute_timestamps: torch.Size([4, 10])
  -> (N_cameras, num_frames)

t0_us: 5100000
clip_id: 25cd4769-5dcf-4b53-a351-bf2c5deb6124
split: train

============================================================
Successfully loaded!
============================================================
"""
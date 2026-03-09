#!/usr/bin/env python3
"""Usage example for LocalPhysicalAIAVDataset.

This script demonstrates how to use the local dataset loader with the
PhysicalAI-Autonomous-Vehicles dataset.

Data location: /home/duzl/hlc/physical_ai_av_usage/data
Note: Only chunk 0 data is available for this demo.
"""

from src.physical_ai_av import LocalPhysicalAIAVDataset, load_physical_aiavdataset
import pandas as pd

DATASET_ROOT = "/home/duzl/hlc/physical_ai_av_usage/data"


def get_chunk_0_clips() -> list[str]:
    """Get all clip IDs from chunk 0 (the only available chunk for demo)."""
    clip_index = pd.read_parquet(f"{DATASET_ROOT}/clip_index.parquet")
    chunk_0_clips = clip_index[clip_index["chunk"] == 0].index.tolist()
    return chunk_0_clips


def example_single_clip():
    """Example 1: Load a single clip using the convenience function."""
    print("=" * 60)
    print("Example 1: Load a single clip")
    print("=" * 60)

    # Use specific clip from chunk 0
    sample_clip_id = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
    print(f"Using clip_id from chunk 0: {sample_clip_id}")

    # Load the clip with multiple cameras
    data = load_physical_aiavdataset(
        DATASET_ROOT,
        sample_clip_id,
        t0_us=5_100_000,  # Start at 5.1 seconds
        num_frames=10,
        cameras=[
            "camera_cross_left_120fov",
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
        ],
    )

    print(f"\nData keys: {list(data.keys())}")
    print(f"image_frames shape: {data['image_frames'].shape}")
    print(f"  -> (num_cameras, num_frames, H, W, C)")
    print(f"ego_history_xyz shape: {data['ego_history_xyz'].shape}")
    print(f"  -> (num_frames, 3) - vehicle positions (x, y, z)")
    print(f"ego_history_rot shape: {data['ego_history_rot'].shape}")
    print(f"  -> (num_frames, 4) - quaternions (qx, qy, qz, qw)")
    print(f"clip_id: {data['clip_id']}")
    print(f"split: {data['split']}")
    print(f"timestamps: {data['timestamps']}")
    print()


def example_dataset_iteration():
    """Example 2: Iterate over chunk 0 clips only."""
    print("=" * 60)
    print("Example 2: Iterate over chunk 0 clips")
    print("=" * 60)

    # Only use clips from chunk 0
    chunk_0_clips = get_chunk_0_clips()
    dataset = LocalPhysicalAIAVDataset(
        DATASET_ROOT,
        clip_ids=chunk_0_clips,
        t0_us=0,
        num_frames=5,
    )

    print(f"Total chunk 0 clips: {len(dataset)}")
    print(f"First 3 clips: {dataset.clip_ids[:3]}")

    # Iterate over first 2 clips
    for i in range(min(2, len(dataset))):
        data = dataset[i]
        print(f"\nClip {i}: {data['clip_id']}")
        print(f"  image_frames shape: {data['image_frames'].shape}")
        print(f"  split: {data['split']}")
    print()


def example_split_filtering():
    """Example 3: Load only a specific split from chunk 0."""
    print("=" * 60)
    print("Example 3: Load only a specific split (chunk 0 only)")
    print("=" * 60)

    # Get chunk 0 clips
    chunk_0_clips = get_chunk_0_clips()

    # Load training split only from chunk 0
    train_dataset = LocalPhysicalAIAVDataset(
        DATASET_ROOT,
        clip_ids=chunk_0_clips,
        split="train",
        t0_us=1_000_000,
        num_frames=10,
    )
    print(f"Train clips in chunk 0: {len(train_dataset)}")

    # Load validation split only from chunk 0
    val_dataset = LocalPhysicalAIAVDataset(
        DATASET_ROOT,
        clip_ids=chunk_0_clips,
        split="val",
        t0_us=1_000_000,
        num_frames=10,
    )
    print(f"Val clips in chunk 0: {len(val_dataset)}")

    # Load test split only from chunk 0
    test_dataset = LocalPhysicalAIAVDataset(
        DATASET_ROOT,
        clip_ids=chunk_0_clips,
        split="test",
        t0_us=1_000_000,
        num_frames=10,
    )
    print(f"Test clips in chunk 0: {len(test_dataset)}")

    # Get a sample from each split
    if len(train_dataset) > 0:
        train_data = train_dataset[0]
        print(f"\nSample train clip: {train_data['clip_id']}, split: {train_data['split']}")

    if len(val_dataset) > 0:
        val_data = val_dataset[0]
        print(f"Sample val clip: {val_data['clip_id']}, split: {val_data['split']}")

    if len(test_dataset) > 0:
        test_data = test_dataset[0]
        print(f"Sample test clip: {test_data['clip_id']}, split: {test_data['split']}")
    print()


def example_multiple_cameras():
    """Example 4: Load multiple cameras (chunk 0 only)."""
    print("=" * 60)
    print("Example 4: Load multiple cameras (chunk 0 only)")
    print("=" * 60)

    # Use specific clip from chunk 0 with all 4 cameras
    dataset = LocalPhysicalAIAVDataset(
        DATASET_ROOT,
        clip_ids=["25cd4769-5dcf-4b53-a351-bf2c5deb6124"],
        cameras=[
            "camera_cross_left_120fov",
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
        ],
        t0_us=2_000_000,
        num_frames=5,
    )

    data = dataset[0]
    print(f"Cameras: {dataset.cameras}")
    print(f"image_frames shape: {data['image_frames'].shape}")
    print(f"  -> (num_cameras={len(dataset.cameras)}, num_frames, H, W, C)")
    print()


def example_with_dataloader():
    """Example 5: Use with PyTorch DataLoader (chunk 0 only)."""
    print("=" * 60)
    print("Example 5: Use with PyTorch DataLoader (chunk 0 only)")
    print("=" * 60)

    from torch.utils.data import DataLoader

    chunk_0_clips = get_chunk_0_clips()
    dataset = LocalPhysicalAIAVDataset(
        DATASET_ROOT,
        clip_ids=chunk_0_clips,
        split="train",
        t0_us=0,
        num_frames=5,
    )

    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Must be 0 due to video reader limitations
    )

    print(f"Dataset size (chunk 0): {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Get one batch
    for batch in dataloader:
        print(f"\nBatch image_frames shape: {batch['image_frames'].shape}")
        print(f"  -> (batch_size, num_cameras, num_frames, H, W, C)")
        print(f"Batch ego_history_xyz shape: {batch['ego_history_xyz'].shape}")
        print(f"  -> (batch_size, num_frames, 3)")
        print(f"Batch clip_ids: {batch['clip_id']}")
        print(f"Batch splits: {batch['split']}")
        break  # Only show first batch
    print()


def example_clip_info():
    """Example 6: Get clip metadata (chunk 0 only)."""
    print("=" * 60)
    print("Example 6: Get clip metadata (chunk 0 only)")
    print("=" * 60)

    chunk_0_clips = get_chunk_0_clips()
    dataset = LocalPhysicalAIAVDataset(DATASET_ROOT, clip_ids=chunk_0_clips)
    clip_id = dataset.clip_ids[0]

    # Get clip info from clip_index
    clip_info = dataset.get_clip_info(clip_id)
    print(f"Clip info for {clip_id}:")
    print(clip_info)

    # Get sensor presence info
    sensor_info = dataset.get_sensor_presence(clip_id)
    print(f"\nSensor presence for {clip_id}:")
    print(sensor_info)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PhysicalAI-AV Local Dataset Usage Examples")
    print(f"Dataset root: {DATASET_ROOT}")
    print("=" * 60 + "\n")

    # Run all examples
    example_single_clip()
    example_dataset_iteration()
    example_split_filtering()
    example_multiple_cameras()
    example_with_dataloader()
    example_clip_info()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)

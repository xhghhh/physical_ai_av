# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import importlib.metadata
import logging

from .utils.hf_interface import HfRepoInterface
from .dataset import PhysicalAIAVDatasetInterface
from .local_dataset import LocalPhysicalAIAVDataset, load_physical_aiavdataset

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = importlib.metadata.version("physical_ai_av")
__all__ = [
    "HfRepoInterface",
    "PhysicalAIAVDatasetInterface",
    "LocalPhysicalAIAVDataset",
    "load_physical_aiavdataset",
]

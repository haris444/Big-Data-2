"""
Timematch Dataset Processing Module

This module provides utilities for loading and processing agricultural parcel time-series data
from the Timematch dataset format. The dataset contains multispectral satellite imagery organized
as Zarr arrays with temporal sequences for crop classification tasks.

Functions:
----------
get_day_of_year_array(date_strings_list: list) -> np.ndarray:
    Converts list of date strings in "YYYYMMDD" format to day-of-year integers.
    Used for temporal positional encoding of the 52 time-series samples.

load_and_analyze_labels(labels_json_path: str, parcel_ids_on_disk: list) -> pd.DataFrame:
    Loads crop type labels from JSON file and filters to only include parcels that exist on disk.
    Returns DataFrame with parcel_id, original_category_label, and count_per_category columns.

filter_and_reindex_labels(labels_df_or_dict, min_samples_per_category: int = 200):
    Filters labels to keep only categories with sufficient samples and reindexes them to 
    consecutive integers starting from 0. Returns filtered DataFrame, mapping dict, and class count.

Classes:
--------
ParcelDataset(Dataset):
    PyTorch Dataset for loading agricultural parcel time-series data from Zarr files with
    fixed pixel sampling to a specified number S.
    
    __init__(parcel_info_list, data_root_path, S_num_sampled_pixels, is_train, 
             num_channels=10, num_time_steps=52, rescale_transform=None, normalization_stats=None):
        - parcel_info_list: List of (parcel_id, new_label_idx) tuples
        - data_root_path: Path to data/ directory containing .zarr files
        - S_num_sampled_pixels: Fixed number of pixels to sample from each parcel
        - is_train: Whether in training mode (enables random pixel sampling vs deterministic)
        - num_channels: Number of spectral channels (default: 10)
        - num_time_steps: Number of time steps (default: 52)
        - rescale_transform: Transform to rescale pixel values (applied before normalization)
        - normalization_stats: Dict with 'mean' and 'std' tensors for channel-wise normalization
    
    __getitem__(idx) -> (torch.Tensor, torch.Tensor):
        Returns (pixel_data_tensor, label_tensor) where pixel_data_tensor has shape 
        (S_num_sampled_pixels, time_steps, channels) and label_tensor is the crop class.

RescalePixels:
    Transform class for rescaling pixel values to [0, 1] range.
    
    __init__(max_pixel_value):
        - max_pixel_value: Maximum expected pixel value for clipping and normalization
    
    __call__(pixel_data) -> torch.Tensor:
        Clamps pixel values to [0, max_pixel_value] and rescales to [0, 1].

Data Format:
------------
- Each parcel is stored as a 3D Zarr array: (52_timepoints, 10_channels, n_pixels)
- Time dimension: 52 fixed acquisition dates per year
- Channel dimension: 10 multispectral bands
- Pixel dimension: Variable number of pixels per parcel
- Labels: Integer crop type categories per parcel
- Dataset samples exactly S pixels from each parcel for consistent batch processing
"""


import numpy as np
from datetime import datetime
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import zarr
import os



def get_day_of_year_array(date_strings_list: list) -> np.ndarray:
   day_of_year_list = []
   for date_str in date_strings_list:
       date_obj = datetime.strptime(date_str, "%Y%m%d")
       day_of_year_list.append(date_obj.timetuple().tm_yday)
   return np.array(day_of_year_list)



def load_and_analyze_labels(labels_json_path: str, parcel_ids_on_disk: list) -> pd.DataFrame:
   with open(labels_json_path, 'r') as f:
       labels_dict = json.load(f)

   # Filter labels to only include parcels that exist on disk
   filtered_data = []
   for parcel_id, category_label in labels_dict.items():
       if parcel_id in parcel_ids_on_disk:
           filtered_data.append({
               'parcel_id': parcel_id,
               'original_category_label': category_label
           })

   df = pd.DataFrame(filtered_data)

   # Add count of samples per category
   category_counts = df['original_category_label'].value_counts().to_dict()
   df['count_per_category'] = df['original_category_label'].map(category_counts)

   return df


def filter_and_reindex_labels(labels_df_or_dict, min_samples_per_category: int = 200):
   if isinstance(labels_df_or_dict, dict):
       df = pd.DataFrame(list(labels_df_or_dict.items()),
                        columns=['parcel_id', 'original_category_label'])
   else:
       df = labels_df_or_dict.copy()

   category_counts = df['original_category_label'].value_counts()
   kept_categories = category_counts[category_counts >= min_samples_per_category].index.tolist()

   filtered_df = df[df['original_category_label'].isin(kept_categories)].copy()

   unique_kept_categories = sorted(filtered_df['original_category_label'].unique())
   category_to_new_index = {cat: idx for idx, cat in enumerate(unique_kept_categories)}

   filtered_df['new_reindexed_label'] = filtered_df['original_category_label'].map(category_to_new_index)

   result_df = filtered_df[['parcel_id', 'new_reindexed_label']].copy()
   index_to_original_mapping = {idx: cat for cat, idx in category_to_new_index.items()}
   num_kept_classes = len(unique_kept_categories)

   return result_df, index_to_original_mapping, num_kept_classes



class ParcelDataset(Dataset):
    def __init__(self, parcel_info_list: list, data_root_path: str, S_num_sampled_pixels: int,
                 is_train: bool, num_channels: int = 10, num_time_steps: int = 52,
                 rescale_transform=None, normalization_stats=None):
        self.parcel_info_list = parcel_info_list
        self.data_root_path = data_root_path
        self.S_num_sampled_pixels = S_num_sampled_pixels
        self.is_train = is_train
        self.num_channels = num_channels
        self.num_time_steps = num_time_steps
        self.rescale_transform = rescale_transform
        self.normalization_stats = normalization_stats

    def __len__(self) -> int:
        return len(self.parcel_info_list)

    def __getitem__(self, idx: int) -> tuple:
        parcel_id, label = self.parcel_info_list[idx]

        # Load pixel data from Zarr
        zarr_array = zarr.open(os.path.join(self.data_root_path, parcel_id + '.zarr'), mode='r')
        N_actual_pixels = zarr_array.shape[2]

        # Pixel sampling logic
        available_indices = np.arange(N_actual_pixels)

        if self.is_train:
            # Training mode: use existing sampling logic with S_num_sampled_pixels
            if N_actual_pixels > self.S_num_sampled_pixels:
                selected_indices = np.random.choice(available_indices, size=self.S_num_sampled_pixels, replace=False)
            elif N_actual_pixels == self.S_num_sampled_pixels:
                selected_indices = available_indices
            else:
                # N_actual_pixels < S_num_sampled_pixels - sample with replacement
                selected_indices = np.random.choice(available_indices, size=self.S_num_sampled_pixels, replace=True)
        else:
            # Evaluation mode: use all available pixels
            selected_indices = available_indices

        # Gather pixel data using selected indices
        sampled_pixel_data = zarr_array[:, :, selected_indices]

        # Data permutation and type conversion
        # From (Time, Channels, num_pixels) to (num_pixels, Time, Channels)
        pixel_data = sampled_pixel_data.transpose(2, 0, 1)
        pixel_data_tensor = torch.tensor(pixel_data, dtype=torch.float32)

        # Apply transforms
        if self.rescale_transform is not None:
            pixel_data_tensor = self.rescale_transform(pixel_data_tensor)

        if self.normalization_stats is not None:
            mean = self.normalization_stats['mean']
            std = self.normalization_stats['std']

            # Keep everything on CPU - no device movement
            pixel_data_tensor = (pixel_data_tensor - mean) / std

        # Create label tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return pixel_data_tensor, label_tensor

class RescalePixels:
   def __init__(self, max_pixel_value):
       self.max_pixel_value = max_pixel_value

   def __call__(self, pixel_data):
       if isinstance(pixel_data, np.ndarray):
           pixel_data = torch.from_numpy(pixel_data)

       pixel_data = pixel_data.to(torch.float32)
       pixel_data = torch.clamp(pixel_data, 0, self.max_pixel_value)
       pixel_data = pixel_data / self.max_pixel_value

       return pixel_data





def calculate_normalization_stats(
    parcel_info_list: list,
    data_root_path: str,
    num_channels: int,
    rescale_transform = None
) -> dict:

    # Initialize accumulators
    channel_sum = np.zeros(num_channels, dtype=np.float64)
    channel_sum_sq = np.zeros(num_channels, dtype=np.float64)
    pixel_count = np.zeros(num_channels, dtype=np.int64)

    # Iterate through each parcel
    for parcel_id, label in parcel_info_list:
        # Construct path to parcel's Zarr file
        parcel_path = os.path.join(data_root_path, parcel_id + '.zarr')

        # Open Zarr array
        zarr_array = zarr.open(parcel_path, mode='r')

        # Load data and convert to float
        parcel_data = np.array(zarr_array, dtype=np.float64)

        # Apply rescale transform if provided
        if rescale_transform is not None:
            parcel_data = rescale_transform(parcel_data)
            # Convert back to numpy if it's a tensor
            if hasattr(parcel_data, 'numpy'):
                parcel_data = parcel_data.numpy()

        # Process each channel
        for c in range(num_channels):
            channel_pixels = parcel_data[:, c, :].flatten()
            channel_sum[c] += np.sum(channel_pixels)
            channel_sum_sq[c] += np.sum(np.square(channel_pixels))
            pixel_count[c] += channel_pixels.size

    # Calculate mean and std
    mean = channel_sum / pixel_count
    std = np.sqrt((channel_sum_sq / pixel_count) - np.square(mean))

    return {'mean': mean, 'std': std}


def collate_fn_padd(batch):
    pixel_data_list = []
    labels_list = []

    # Extract pixel data and labels from batch
    for pixel_data, label in batch:
        pixel_data_list.append(pixel_data)
        labels_list.append(label)

    # Find maximum number of pixels
    max_S = max(pixel_data.shape[0] for pixel_data in pixel_data_list)

    # Get dimensions
    T = pixel_data_list[0].shape[1]  # time steps
    C = pixel_data_list[0].shape[2]  # channels
    B = len(pixel_data_list)         # batch size

    # Initialize padded tensor and mask
    padded_pixel_data = torch.zeros(B, max_S, T, C, dtype=pixel_data_list[0].dtype)
    mask = torch.zeros(B, max_S, dtype=torch.bool)

    # Fill padded tensor and create mask
    for i, pixel_data in enumerate(pixel_data_list):
        S = pixel_data.shape[0]  # actual number of pixels
        padded_pixel_data[i, :S] = pixel_data
        mask[i, :S] = True

    # Stack labels
    labels = torch.stack(labels_list)

    return padded_pixel_data, mask, labels
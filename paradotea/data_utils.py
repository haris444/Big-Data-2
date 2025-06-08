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

   #filter labels
   filtered_data = []
   for parcel_id, category_label in labels_dict.items():
       if parcel_id in parcel_ids_on_disk:
           filtered_data.append({
               'parcel_id': parcel_id,
               'original_category_label': category_label
           })

   df = pd.DataFrame(filtered_data)

   #samples / category
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

        # Load pixel data from zarrs
        zarr_array = zarr.open(os.path.join(self.data_root_path, parcel_id + '.zarr'), mode='r')
        N_actual_pixels = zarr_array.shape[2]

        #sample
        available_indices = np.arange(N_actual_pixels)

        if self.is_train:
            #if training then sample
            if N_actual_pixels > self.S_num_sampled_pixels:
                selected_indices = np.random.choice(available_indices, size=self.S_num_sampled_pixels, replace=False)
            elif N_actual_pixels == self.S_num_sampled_pixels:
                selected_indices = available_indices
            else:
                selected_indices = np.random.choice(available_indices, size=self.S_num_sampled_pixels, replace=True)
        else:
            #if eval use all pixels
            selected_indices = available_indices

        sampled_pixel_data = zarr_array[:, :, selected_indices]

        #(T, C, N) to (N, T, C)
        pixel_data = sampled_pixel_data.transpose(2, 0, 1)
        pixel_data_tensor = torch.tensor(pixel_data, dtype=torch.float32)

        # Apply transforms
        if self.rescale_transform is not None:
            pixel_data_tensor = self.rescale_transform(pixel_data_tensor)

        if self.normalization_stats is not None:
            mean = self.normalization_stats['mean']
            std = self.normalization_stats['std']

            pixel_data_tensor = (pixel_data_tensor - mean) / std

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

    channel_sum = np.zeros(num_channels, dtype=np.float64)
    channel_sum_sq = np.zeros(num_channels, dtype=np.float64)
    pixel_count = np.zeros(num_channels, dtype=np.int64)

    for parcel_id, label in parcel_info_list:

        parcel_path = os.path.join(data_root_path, parcel_id + '.zarr')

        zarr_array = zarr.open(parcel_path, mode='r')

        parcel_data = np.array(zarr_array, dtype=np.float64)

        if rescale_transform is not None:
            parcel_data = rescale_transform(parcel_data)
            if hasattr(parcel_data, 'numpy'):
                parcel_data = parcel_data.numpy()

        for c in range(num_channels):
            channel_pixels = parcel_data[:, c, :].flatten()
            channel_sum[c] += np.sum(channel_pixels)
            channel_sum_sq[c] += np.sum(np.square(channel_pixels))
            pixel_count[c] += channel_pixels.size

    #mean and std
    mean = channel_sum / pixel_count
    std = np.sqrt((channel_sum_sq / pixel_count) - np.square(mean))

    return {'mean': mean, 'std': std}


def collate_fn_padd(batch):
    pixel_data_list = []
    labels_list = []

    # pixel data and labels from batch
    for pixel_data, label in batch:
        pixel_data_list.append(pixel_data)
        labels_list.append(label)

    #max num pixels
    max_S = max(pixel_data.shape[0] for pixel_data in pixel_data_list)

    T = pixel_data_list[0].shape[1]  #T
    C = pixel_data_list[0].shape[2]  #C
    B = len(pixel_data_list)         #B

    padded_pixel_data = torch.zeros(B, max_S, T, C, dtype=pixel_data_list[0].dtype)
    mask = torch.zeros(B, max_S, dtype=torch.bool)

    for i, pixel_data in enumerate(pixel_data_list):
        S = pixel_data.shape[0]
        padded_pixel_data[i, :S] = pixel_data
        mask[i, :S] = True

    labels = torch.stack(labels_list)

    return padded_pixel_data, mask, labels
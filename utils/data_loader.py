import os
from typing import List
import numpy as np
from pathlib import Path
import pypianoroll
import random
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset
)
from tqdm import tqdm

from config import (
    batch_size,
    beat_resolution,
    data_dir,
    dataset_output_dir,
    lowest_pitch,
    measure_resolution,
    music_groups,
    n_measures,
    n_pitches,
    n_samples_per_song
)

class CreateDataLoader:
    def __init__(self, data_dir: str = data_dir, music_groups: str = music_groups):
        self.dataset_root = Path(data_dir)
        self.music_groups = music_groups
    
    def msd_id_to_dirs(self, msd_id: str) -> str:
        """Given an MSD ID, generate the path prefix.
        E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
        return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

    def _get_id_list(self) -> List[str]:
        id_list = []
        for path in os.listdir(music_groups):
            filepath = os.path.join(music_groups, path)
            if os.path.isfile(filepath):
                with open(filepath) as f:
                    id_list.extend([line.rstrip() for line in f])
        id_list = list(set(id_list))
        return id_list

    def _get_data(self, ids: List[str]) -> list:
        data = []
        for msd_id in tqdm(ids):
            song_dir = self.dataset_root / self.msd_id_to_dirs(msd_id)
            multitrack = pypianoroll.load(song_dir / os.listdir(song_dir)[0])
            # Binarize the pianorolls
            multitrack.binarize()
            # Downsample the pianorolls (shape: n_timesteps x n_pitches)
            multitrack.set_resolution(beat_resolution)
            # Stack the pianoroll (shape: n_tracks x n_timesteps x n_pitches)
            pianoroll = (multitrack.stack() > 0)
            # Get the target pitch range only
            pianoroll = pianoroll[:, :, lowest_pitch:lowest_pitch + n_pitches]
            # Calculate the total measures
            n_total_measures = multitrack.get_max_length() // measure_resolution
            candidate = n_total_measures - n_measures
            target_n_samples = min(n_total_measures // n_measures, n_samples_per_song)
            # Randomly select a number of phrases from the multitrack pianoroll
            for idx in np.random.choice(candidate, target_n_samples, False):
                start = idx * measure_resolution
                end = (idx + n_measures) * measure_resolution
                # Skip the samples where some track(s) has too few notes
                if (pianoroll.sum(axis=(1, 2)) < 10).any():
                    continue
                data.append(pianoroll[:, start:end])
        # Stack all the collected pianoroll segments into one big array
        random.shuffle(data)
        data = np.stack(data)
        return data

    def _create_data_loader(self) -> DataLoader:
        ids = self._get_id_list()
        data = self._get_data(ids)
        data = torch.as_tensor(data, dtype=torch.float32)
        # saving the dataset
        torch.save(data, dataset_output_dir)
        dataset = TensorDataset(data)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        return data_loader

    def _load_data_loader(self) -> DataLoader:
        data = torch.load(dataset_output_dir)
        dataset = TensorDataset(data)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        return data_loader

    def load_data_loader(self) -> DataLoader:
        if os.path.exists(dataset_output_dir):
            return self._load_data_loader()
        return self._create_data_loader()

if __name__ == "__main__":
    create_data_loader = CreateDataLoader()
    data_loader = create_data_loader.load_data_loader()
    print(f"Length of data loader: {len(data_loader)};")
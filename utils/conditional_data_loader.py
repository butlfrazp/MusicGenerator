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

import json

from config import (
    batch_size,
    beat_resolution,
    data_dir,
    conditional_dataset_output_dir,
    conditional_dataset_label_output_dir,
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
        
        with open('./data/class_map.json', 'r') as f:
            json_data = f.read().encode('utf-8')
            self.class_map = json.loads(json_data)
            self.classes = list(set(self.class_map.values()))
            self.one_hot_encoded_keys = { c: i for i, c in enumerate(self.classes) }
    
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
        labels = []
        for msd_id in tqdm(ids):
            # is the msg id not in map move on
            if msd_id not in self.class_map:
                continue

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
                
                pianoroll_class = self.class_map[msd_id]
                one_hot = torch.zeros(len(self.classes))
                one_hot[self.one_hot_encoded_keys[pianoroll_class]] = 1
                labels.append(one_hot)
                data.append(pianoroll[:, start:end])
        # Stack all the collected pianoroll segments into one big array
        shuf = list(zip(data, labels))
        random.shuffle(shuf)
        data, labels = zip(*shuf)
        labels = np.stack(labels)
        data = np.stack(data)
        return data, labels

    def _create_data_loader(self) -> DataLoader:
        ids = self._get_id_list()
        data, labels = self._get_data(ids)
        data = torch.as_tensor(data, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        # saving the dataset
        torch.save(data, conditional_dataset_output_dir)
        torch.save(labels, conditional_dataset_label_output_dir)
        dataset = TensorDataset(data, labels)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        return data_loader

    def _load_data_loader(self) -> DataLoader:
        data = torch.load(conditional_dataset_output_dir)
        labels = torch.load(conditional_dataset_label_output_dir)
        dataset = TensorDataset(data, labels)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        return data_loader

    def load_data_loader(self) -> DataLoader:
        if os.path.exists(conditional_dataset_output_dir) and os.path.exists(conditional_dataset_label_output_dir):
            return self._load_data_loader()
        return self._create_data_loader()

if __name__ == "__main__":
    create_data_loader = CreateDataLoader()
    data_loader = create_data_loader.load_data_loader()
    # print(f"Length of data loader: {len(data_loader)};")
    data, label = next(iter(data_loader))

import configparser as cp
import logging
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

viirs_batch = namedtuple('viirs_batch', ['viirs', 'diurnal'])


class WildFireDataset(Dataset):
    def __init__(self, train=True, load_records=None, config_file="config.ini"):
        config_file = Path(config_file)
        config = cp.ConfigParser()
        if not config_file.is_file():
            raise FileNotFoundError(f"{config_file} does not exist")
        config.read(config_file)
        if train:
            self.source_file = config["DEFAULT"]["train_set"]
        else:  # testing
            self.source_file = config["DEFAULT"]["test_set"]
        logging.info(f"Data source file: {self.source_file}")

        with h5py.File(self.source_file, 'r') as h5_ptr:
            # ['datetime', 'land_cover', 'latitude', 'longitude', 'meteorology', 'observed', 'target']
            fields = list(h5_ptr)

            self.data = {}
            for f in fields:
                if isinstance(load_records, int):
                    self.data[f] = h5_ptr[f][:load_records]
                else:  # list or np.array
                    self.original_indexes = list(load_records)
                    self.data[f] = h5_ptr[f][self.original_indexes]

    def get_by_original_indexes(self, index):
        assert isinstance(index, int)
        item = self.original_indexes.index(index)
        return self.data['observed'][item, ...].astype(np.float32), self.data['target'][item, ...].astype(np.float32)

    def __len__(self):
        return len(self.data['observed'])

    def __getitem__(self, item):
        return self.data['observed'][item, ...].astype(np.float32), self.data['target'][item, ...].astype(np.float32)


class VIIRSDataset(Dataset):
    def __init__(self, train=True, load_records=None, config_file="config.ini"):
        config_file = Path(config_file)
        config = cp.ConfigParser()
        if not config_file.is_file():
            raise FileNotFoundError(f"{config_file} does not exist")
        config.read(config_file)
        if train:
            self.source_file = config["DEFAULT"]["train_set"]
        else:  # testing
            self.source_file = config["DEFAULT"]["test_set"]
        logging.info(f"Data source file: {self.source_file}")

        with h5py.File(self.source_file, 'r') as h5_ptr:
            # ['datetime', 'land_cover', 'latitude', 'longitude', 'meteorology', 'observed', 'target']
            indexes = np.arange(len(h5_ptr['observed']))
            np.random.shuffle(indexes)
            indexes = indexes[:load_records]
            indexes.sort()
            fields = list(h5_ptr)
            self.data = {"timestep_observed": np.tile([0, -1, -2, -3, -4], load_records).astype(int),
                         "timestep_target": np.tile([1, 2], load_records).astype(int)}

            for f in fields:
                self.data[f] = h5_ptr[f][indexes]
                if f in ["target", "observed"]:
                    self.data[f] = self.data[f].reshape(-1, 30, 30)
                if f == "datetime":
                    self.data["datetime"] = pd.to_datetime(self.data["datetime"])

        self.data["viirs"] = np.concatenate((self.data["observed"], self.data["target"]), axis=0)
        self.data["timestep"] = np.concatenate((self.data["timestep_observed"], self.data["timestep_target"]), axis=0)
        self.data["indexes"] = np.concatenate((np.repeat(indexes, 5), np.repeat(indexes, 2))).astype(int)
        self.data["hours"] = self.data["datetime"].hour
        observed_diurnality = np.zeros((load_records, 5), dtype=int)
        observed_diurnality[:, 0] = (self.data["hours"] <= 12).astype(int)

        observed_diurnality[:, 1] = 1 - observed_diurnality[:, 0]
        observed_diurnality[:, 2] = 1 - observed_diurnality[:, 1]
        observed_diurnality[:, 3] = 1 - observed_diurnality[:, 2]
        observed_diurnality[:, 4] = 1 - observed_diurnality[:, 3]

        target_diurnality = np.zeros((load_records, 2), dtype=int)
        target_diurnality[:, 0] = 1 - observed_diurnality[:, 0]
        target_diurnality[:, 1] = 1 - target_diurnality[:, 0]
        self.data["diurnal"] = np.concatenate((observed_diurnality.flatten(), target_diurnality.flatten())).astype(
            np.float32)
        unique, count = np.unique(self.data["diurnal"], return_counts=True)
        count = count / np.sum(count)
        print({f"{u}: {c:.3f}" for u, c in zip(unique, count)})

    def __len__(self):
        return len(self.data['viirs'])

    def __getitem__(self, item) -> viirs_batch:
        return viirs_batch(viirs=self.data['viirs'][item, ...].astype(np.float32), diurnal=self.data['diurnal'][item])

    @property
    def indexes(self):
        return self.data["indexes"].astype(np.int)

    @property
    def x(self):
        return self.data['viirs'].astype(np.float32)

    @property
    def diurnal(self):
        return self.data['diurnal']

    @property
    def timestep(self):
        return self.data['timestep'].astype(np.float32)

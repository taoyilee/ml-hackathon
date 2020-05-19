import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py

class WildFireData(Dataset):
    def __init__(self, data_path, target_hour, obsv_tagt_transforms, land_transforms, meteo_transforms):
        self.data = {}
        with h5py.File(data_path, 'r') as f:
            for k in list(f):
                self.data[k] = f[k][:]

        limit = 5000
        self.data['observed'] = self.data['observed'][:limit]
        self.data['target'] = self.data['target'][:limit]
        self.data['land_cover'] = self.data['land_cover'][:limit]
        self.data['meteorology'] = self.data['meteorology'][:limit]

        self.obsv_tagt_transforms = obsv_tagt_transforms
        self.land_transforms = land_transforms
        self.meteo_transforms = meteo_transforms
        self.target_hour = target_hour

    def __len__(self):
        return self.data['observed'].shape[0]

    def __getitem__(self, idx):
        sample_observed = self.data['observed'][idx]
        if self.target_hour == 12:
            sample_target = self.data['target'][idx, 0][np.newaxis, :]
        else:
            sample_target = self.data['target'][idx, 1][np.newaxis, :]
            sample_observed = np.concat([sample_observed, self.data['target'][idx, 0]])

        sample_land_cover = self.data['land_cover'][idx]
        sample_meteorology = self.data['meteorology'][idx]
        sample_meteorology = sample_meteorology.reshape(-1, sample_meteorology.shape[2], sample_meteorology.shape[3])

        sample_observed = np.nan_to_num(sample_observed)
        sample_target = np.nan_to_num(sample_target)
        sample_land_cover = np.nan_to_num(sample_land_cover)
        sample_meteorology = np.nan_to_num(sample_meteorology)

        sample_observed = np.moveaxis(np.nan_to_num(sample_observed), 0, -1)
        sample_target = np.moveaxis(np.nan_to_num(sample_target), 0, -1)
        sample_land_cover = np.moveaxis(sample_land_cover, 0, -1)
        sample_meteorology = np.moveaxis(sample_meteorology, 0, -1)

        sample_observed = self.obsv_tagt_transforms(sample_observed)
        sample_target = self.obsv_tagt_transforms(sample_target)
        sample_land_cover = self.land_transforms(sample_land_cover)
        sample_meteorology = self.meteo_transforms(sample_meteorology)

        X = torch.cat([sample_observed, sample_land_cover, sample_meteorology.float()], dim=0)
        y = sample_target

        return X, y


def get_wild_fire_data(data_path, target_hour):
    obsv_tagt_transforms = transforms.Compose([transforms.ToTensor()])
    land_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5] * 17, std=[.5] * 17)])
    meteo_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5] * 10, std=[.5] * 10)])

    return WildFireData(data_path, target_hour, obsv_tagt_transforms, land_transforms, meteo_transforms)


def get_data_loader(batch_size, target_hour):
    train_data_path = 'uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train.hdf5'
    train_data = get_wild_fire_data(train_data_path, target_hour)

    test_data_path = 'uci_ml_hackathon_fire_dataset_2013-01-01_2014-01-01_5k_test.hdf5'
    test_data = get_wild_fire_data(test_data_path, target_hour)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader

import h5py
import numpy as np
import pandas as pd
if __name__ == "__main__":
    with h5py.File("data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_30k_train_v2.hdf5", "r") as h5_ptr:
        for f in h5_ptr.keys():
            if f == "datetime":
                time = pd.to_datetime(h5_ptr[f][:])
                print(f, h5_ptr[f].shape, time.min(), time.max())

            elif f == "land_cover":
                print(f, h5_ptr[f].shape)
                for i in range(17):
                    print(f"\t{i}, {np.nanmin(h5_ptr[f][:, i, :, :]):.2f}, {np.nanmax(h5_ptr[f][:, i, :, :]):.2f}")
            elif f == "meteorology":
                print(f, h5_ptr[f].shape)
                for i in range(5):
                    print(f"\t{i}, {np.nanmin(h5_ptr[f][:, :, i, :, :]):.2e}, {np.nanmax(h5_ptr[f][:, :, i, :, :]):.2e}")

            else:
                print(f"{f}, {h5_ptr[f].shape}, {h5_ptr[f][:].min():.2f}, {h5_ptr[f][:].max():.2f}")

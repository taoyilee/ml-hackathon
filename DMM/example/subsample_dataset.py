import h5py
import numpy as np

if __name__ == "__main__":
    n = 1000
    np.random.seed(0)

    with h5py.File("data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_30k_train_v2.hdf5", "r") as h5_ptr:
        print(h5_ptr.keys())
        indexes = np.arange(h5_ptr['datetime'].shape[0])
        np.random.shuffle(indexes)
        indexes = indexes[:n]
        indexes.sort()

        with h5py.File("data/dataset_1k.hdf5", "w") as write_ptr:
            for f in h5_ptr.keys():
                write_ptr[f] = h5_ptr[f][indexes]
            write_ptr["original_index"] = indexes

    with h5py.File("data/dataset_1k.hdf5", "r") as h5_ptr:
        for f in h5_ptr.keys():
            if f == "land_cover":
                land_cover = h5_ptr['land_cover']
                land_cover = np.nan_to_num(land_cover)
                for j in range(land_cover.shape[1]):
                    print(f"LC-{j}", land_cover[:, j, ...].shape, land_cover[:, j, ...].min(), land_cover[j].max())
                # land_cover_mask = ~np.isnan(land_cover)
                # print(f, h5_ptr[f].shape, h5_ptr[f][land_cover_mask].max(), h5_ptr[f][land_cover_mask].min())
            else:
                print(f, h5_ptr[f].shape, h5_ptr[f][:].max(), h5_ptr[f][:].min())

import numpy as np
import pytest
from torch.utils.data import Dataset

from dataset import VIIRSDataset, WildFireDataset


@pytest.mark.slow
def test_data_loader():
    data_loader = VIIRSDataset()
    assert data_loader.source_file == "data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train.hdf5"
    data_loader = VIIRSDataset(train=True)
    assert data_loader.source_file == "data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train.hdf5"
    data_loader = VIIRSDataset(train=False)
    assert data_loader.source_file == "data/uci_ml_hackathon_fire_dataset_2013-01-01_2014-01-01_5k_test.hdf5"


@pytest.mark.parametrize("load_records", range(1, 10))
def test_data_lenth(load_records):
    data_loader = VIIRSDataset(load_records=load_records)
    assert len(data_loader) == 7 * load_records


@pytest.mark.parametrize("load_records,idx", zip(range(1, 10), range(0, 9)))
def test_get_item(load_records, idx):
    data_loader = VIIRSDataset(load_records=load_records)
    assert data_loader[idx].shape == (30, 30)


def test_correct_reshape():
    n = 40
    viirs_data_loader = VIIRSDataset(load_records=n)
    wildfire_data_loader = WildFireDataset(load_records=n)

    expected = np.concatenate((wildfire_data_loader[:][0].reshape(-1, 30, 30),
                               wildfire_data_loader[:][1].reshape(-1, 30, 30))
                              , axis=0)
    assert np.array_equal(np.array(viirs_data_loader[:]), expected)


@pytest.mark.parametrize("idx", [0, 5])
def test_non_zero(idx):
    viirs_data_loader = VIIRSDataset(load_records=100)
    assert np.any(np.array(viirs_data_loader[idx]))


@pytest.mark.parametrize("idx", [1, 2, 3, 4])
def test_zero(idx):
    viirs_data_loader = VIIRSDataset(load_records=100)
    assert np.all(np.array(viirs_data_loader[idx]) == 0)


def test_dtype():
    viirs_data_loader = VIIRSDataset(load_records=100)
    assert viirs_data_loader[0].dtype == np.float32


def test_x_shape():
    viirs_data_loader = VIIRSDataset(load_records=100)
    assert viirs_data_loader.x.shape == (7 * 100, 30, 30)


def test_timestep_shape():
    viirs_data_loader = VIIRSDataset(load_records=100)
    assert viirs_data_loader.timestep.shape == (7 * 100,)


def test_indexes():
    viirs_data_loader = VIIRSDataset(load_records=5)
    assert np.array_equal(viirs_data_loader.indexes,
                          np.array([0, 0, 0, 0, 0,
                                    1, 1, 1, 1, 1,
                                    2, 2, 2, 2, 2,
                                    3, 3, 3, 3, 3,
                                    4, 4, 4, 4, 4,
                                    0, 0, 1, 1, 2, 2, 3, 3, 4, 4
                                    ]).astype(int))


def test_timestep():
    viirs_data_loader = VIIRSDataset(load_records=5)
    assert np.array_equal(viirs_data_loader.timestep,
                          np.concatenate((np.array(
                              [0, -1, -2, -3, -4,
                               0, -1, -2, -3, -4,
                               0, -1, -2, -3, -4,
                               0, -1, -2, -3, -4,
                               0, -1, -2, -3, -4]),
                                          np.tile([1, 2], 5))).astype(int))
    viirs_data_loader = VIIRSDataset(load_records=6)
    assert np.array_equal(viirs_data_loader.timestep,
                          np.concatenate((np.tile([0, -1, -2, -3, -4], 6),
                                          np.tile([1, 2], 6))).astype(int))


@pytest.mark.xfail(reason="test_zero xfail")
@pytest.mark.parametrize("idx", [0, 5])
def test_zero_xfail(idx):
    viirs_data_loader = VIIRSDataset(load_records=100)
    assert np.all(np.array(viirs_data_loader[idx]) == 0)


def test_config_not_found():
    with pytest.raises(FileNotFoundError):
        VIIRSDataset(config_file="nonexistent.ini")


@pytest.mark.slow
def test_is_dataset():
    assert isinstance(VIIRSDataset(), Dataset)

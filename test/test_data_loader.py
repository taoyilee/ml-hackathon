import pytest
from torch.utils.data import Dataset

from src.data.dataset import WildFireDataset


@pytest.mark.slow
def test_data_loader():
    data_loader = WildFireDataset()
    assert data_loader.source_file == "data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train.hdf5"
    data_loader = WildFireDataset(train=True)
    assert data_loader.source_file == "data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train.hdf5"
    data_loader = WildFireDataset(train=False)
    assert data_loader.source_file == "data/uci_ml_hackathon_fire_dataset_2013-01-01_2014-01-01_5k_test.hdf5"


@pytest.mark.parametrize("l", range(1, 10))
def test_data_lenth(l):
    data_loader = WildFireDataset(load_records=l)
    assert len(data_loader) == l


@pytest.mark.parametrize("load_records,idx", zip(range(1, 10), range(0, 9)))
def test_get_item(load_records, idx):
    data_loader = WildFireDataset(load_records=load_records)
    assert data_loader[idx][0].shape == (5, 30, 30)
    assert data_loader[idx][1].shape == (2, 30, 30)


def test_config_not_found():
    with pytest.raises(FileNotFoundError):
        data_loader = WildFireDataset(config_file="nonexistent.ini")


@pytest.mark.slow
def test_is_dataset():
    assert isinstance(WildFireDataset(), Dataset)

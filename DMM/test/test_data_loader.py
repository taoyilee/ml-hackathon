import pytest
from torch.utils.data import Dataset

from src.data import WildFireDataset


@pytest.mark.slow
def test_data_loader():
    data_loader = WildFireDataset()
    assert data_loader.source_file == "data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train.hdf5"
    data_loader = WildFireDataset(train=True)
    assert data_loader.source_file == "data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train.hdf5"
    data_loader = WildFireDataset(train=False)
    assert data_loader.source_file == "data/uci_ml_hackathon_fire_dataset_2013-01-01_2014-01-01_5k_test.hdf5"


def test_data_lenth():
    data_loader = WildFireDataset()
    assert len(data_loader) == 10000


@pytest.mark.parametrize("idx", list(range(0, 9)))
def test_get_item(idx):
    data_loader = WildFireDataset()
    assert data_loader[idx].viirs.shape == (1, 7, 30, 30)
    assert data_loader[idx].diurnality.shape == (1,)

    assert data_loader[idx:idx + 3].viirs.shape == (3, 7, 30, 30)
    assert data_loader[idx:idx + 3].diurnality.shape == (3,)


def test_config_not_found():
    with pytest.raises(FileNotFoundError):
        data_loader = WildFireDataset(config_file="nonexistent.ini")


@pytest.mark.slow
def test_is_dataset():
    assert isinstance(WildFireDataset(), Dataset)

import configparser as cp
import json
from pathlib import Path

import numpy as np
import torch

from src import logger
from src.data import WildFireDataset
from src.vae import VAE, VAEConfig
from src.visualize.vae_plots import plot_tsne, plot_latent, plot_epoch, plot_forecast


def eval_dmm(experiment_dir):
    experiment_dir = Path(experiment_dir)
    config = cp.ConfigParser()
    config.read(experiment_dir / "config.ini")
    with open(experiment_dir / "metrics.json", "rb") as fptr:
        metrics = json.load(fptr)

    # load dataset
    logger.info(f"Loading dataset")
    batch_size = config["vae-eval"].getint("batch_size")

    wildfire_dataset = WildFireDataset(train=True, config_file=experiment_dir / "config.ini")
    from torch.utils.data import DataLoader
    data_loader = DataLoader(wildfire_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    plot_epoch(experiment_dir, -np.array(metrics['elbo']['values']), "ELBO", ylim=(-10, 0))
    for f in ['alpha', 'beta', 'inferred_mean', 'inferred_std']:
        plot_epoch(experiment_dir, metrics[f]['values'], f)

    logger.info(f"Loading model")
    with open(experiment_dir / "vae_config.json", "rb") as fptr:
        vae_config = json.load(fptr, object_hook=lambda dct: VAEConfig(**dct))  # type:VAEConfig

    vae = VAE(vae_config)
    vae.load_state_dict(torch.load(experiment_dir / "model_final.pt"))

    z_loc, _ = get_latent(vae, data_loader)

    plot_tsne(z_loc, experiment_dir, wildfire_dataset)
    plot_latent(z_loc, experiment_dir, wildfire_dataset)
    plot_forecast(vae, experiment_dir, wildfire_dataset, data_loader)


def get_latent(vae: "VAE", data_loader):
    from src.data.dataset import _ct

    z_loc, z_scale = None, None
    logger.info(f"Encoding observation into latent space")
    with torch.no_grad():
        for d in data_loader:
            if z_loc is None:
                z_loc, z_scale = vae.encode(_ct(d.diurnality), _ct(d.viirs))
            else:
                z_loc_i, z_scale_i = vae.encode(_ct(d.diurnality), _ct(d.viirs))
                z_loc = np.concatenate((z_loc, z_loc_i), axis=1)
                z_scale = np.concatenate((z_scale, z_scale_i), axis=1)

    return z_loc.swapaxes(0, 1), z_scale.swapaxes(0, 1)

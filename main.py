import configparser as cp
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO

from src.data.wildfire import WildFireData
from src.utils import beta_to_mean_std
from src.vae import VAE
from src.visualize.vae_plots import plot_observation, plot_tsne, plot_latent, plot_elbo

if __name__ == "__main__":

    config = cp.ConfigParser()
    config.read("config.ini")
    epochs = config["vae-training"].getint("epochs")
    z_dim = config["vae-training"].getint("z_dim")

    n_samples = config["vae-training"].getint("n_samples")
    pyro.clear_param_store()

    # Seed randomness for repeatability
    pyro.set_rng_seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    vae_results = Path(config["DEFAULT"]["vae_results"])
    vae_results.mkdir(exist_ok=True)

    wildfire_dataset = WildFireData(config["DEFAULT"]["train_set"], n_samples)
    vae = VAE(z_dim=z_dim)

    svi = SVI(vae.model, vae.guide, vae.optimizer, loss=Trace_ELBO())
    elbo_loss_log = []
    for step in range(epochs):
        elbo_loss_log.append(svi.step(wildfire_dataset.diurnality, wildfire_dataset.observed_0) / n_samples)
        if (step + 1) % 200 == 0:
            print(f"[Iter {step + 1:05d}] ELBO: {elbo_loss_log[-1] :.3f}")

    alpha_q = pyro.param("alpha_q").item()
    beta_q = pyro.param("beta_q").item()
    inferred_mean, inferred_std = beta_to_mean_std(alpha_q, beta_q)

    print(f"Diurnality estimate {inferred_mean:.2f}+/-{inferred_std:.2f}. alpha = {alpha_q:.2f}; beta = {beta_q:.2f} ")
    plot_elbo(vae_results, elbo_loss_log)
    plot_tsne(vae, vae_results, wildfire_dataset)
    plot_latent(vae, vae_results, wildfire_dataset)

    if config["visualization"].getboolean("plot_observation"):
        observed_0 = wildfire_dataset.observed_0.cpu().numpy()
        diurnality = wildfire_dataset.diurnality.cpu().numpy()
        plot_observation(vae_results, wildfire_dataset.indexes[diurnality == 1],
                         observed_0[diurnality == 1], "day", 10)
        plot_observation(vae_results, wildfire_dataset.indexes[diurnality == 0],
                         observed_0[diurnality == 0], "night", 10)

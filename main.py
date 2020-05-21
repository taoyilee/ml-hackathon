import configparser as cp
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO

from src.data.wildfire import WildFireData
from src.utils import beta_to_mean_std
from src.vae import VAE
from src.visualize.vae_plots import plot_observation, plot_tsne, plot_latent, plot_epoch, plot_forecast

if __name__ == "__main__":

    config = cp.ConfigParser()
    config.read("config.ini")
    epochs = config["vae-training"].getint("epochs")
    z_dim = config["vae-training"].getint("z_dim")
    min_af = config["vae-training"].getfloat("min_af")
    annealing_epochs = config["vae-training"].getint("annealing_epochs")
    n_samples = config["vae-training"].getint("n_samples")
    pyro.clear_param_store()

    # Seed randomness for repeatability
    pyro.set_rng_seed(0)
    torch.manual_seed(0)
    # TODO: making cuDNN deterministic seems to have no effect ?? Dropout layer will be different each time
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    np.random.seed(0)

    vae_results = Path(config["DEFAULT"]["vae_results"])
    vae_results.mkdir(exist_ok=True)

    wildfire_dataset = WildFireData(config["DEFAULT"]["train_set"], n_samples)
    vae = VAE(z_dim=z_dim)

    svi = SVI(vae.model, vae.guide, vae.optimizer, loss=Trace_ELBO())
    elbo_loss_log = []
    alpha = []
    beta = []
    mean = []
    std = []

    for step in range(epochs):
        if step < annealing_epochs:
            annealing_factor = min_af + (1.0 - min_af) * step / annealing_epochs
        else:
            annealing_factor = 1.0
        mean_loss = svi.step(wildfire_dataset.diurnality, wildfire_dataset.viirs, annealing_factor) / n_samples

        elbo_loss_log.append(mean_loss)
        alpha.append(pyro.param("alpha").item())
        beta.append(pyro.param("beta").item())
        if (step + 1) % 200 == 0:
            inferred_mean, inferred_std = beta_to_mean_std(alpha[-1], beta[-1])
            mean.append(inferred_mean)
            std.append(inferred_std)
            print(f"[Iter {step + 1:05d}] af = {annealing_factor:.2f} ELBO: {elbo_loss_log[-1] :.3f}"
                  f" diurnality: {inferred_mean * 100:.2f} +/- {100 * inferred_std:.2f} %"
                  f" alpha = {alpha[-1]:.2f}; beta = {beta[-1]:.2f} ")

    torch.save(vae.state_dict(), vae_results / "model_final.pt")
    vae.optimizer.save(vae_results / "optimizer.pt")

    plot_epoch(vae_results, -np.array(elbo_loss_log), "ELBO", ylim=(-1000, 0))
    plot_epoch(vae_results, alpha, "alpha")
    plot_epoch(vae_results, beta, "beta")
    plot_epoch(vae_results, mean, "mean")
    plot_epoch(vae_results, std, "std")
    plot_tsne(vae, vae_results, wildfire_dataset)
    plot_latent(vae, vae_results, wildfire_dataset)
    plot_forecast(vae, vae_results, wildfire_dataset)

    if config["visualization"].getboolean("plot_observation"):
        viirs = wildfire_dataset.viirs.cpu().numpy()
        diurnality = wildfire_dataset.diurnality.cpu().numpy()
        plot_observation(vae_results, wildfire_dataset.indexes[diurnality == 1], viirs[diurnality == 1], "day", 10)
        plot_observation(vae_results, wildfire_dataset.indexes[diurnality == 0], viirs[diurnality == 0], "night", 10)

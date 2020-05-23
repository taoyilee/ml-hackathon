# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from matplotlib import pyplot as plt

from src import logger
from src.data import WildFireDataset
from src.vae import VAE

mapping = {0: -48, 1: -36, 2: -24, 3: -12, 4: 0, 5: 12, 6: 24}

import torch


def plot_observation(vae_results, indexes, observation, label, n=10):
    plt.figure(figsize=(n * 1.8, n * 1.8))
    fn = 1
    counter = 0
    for i, (idx, obs) in enumerate(zip(indexes, observation)):
        plt.subplot(n, n, counter + 1)
        plt.title(f'# {idx}')
        plt.imshow(obs)
        plt.axis('off')

        counter += 1
        if counter == n ** 2:
            plt.tight_layout()
            plt.savefig(vae_results / f"viirs_{label}_{fn:05d}.png")
            plt.close()
            fn += 1
            counter = 0
            plt.figure(figsize=(n * 1.8, n * 1.8))


def plot_latent(z_loc, vae_results, wildfire_dataset: "WildFireDataset"):
    t_max = z_loc.shape[1]
    features = z_loc.shape[2]
    logger.info(f"Plotting laetent space")

    diurnality = wildfire_dataset[:].diurnality
    for time_step in range(t_max):
        day = z_loc[diurnality == 1, time_step, :]
        night = z_loc[diurnality == 0, time_step, :]
        plt.figure(figsize=(21, 6))
        for j in range(features):
            plt.subplot(1, features, j + 1)
            plt.hist(day[:, j], color=f"C0", label="day")
            plt.hist(night[:, j], color=f"C1", label="night")
            plt.legend(loc=1)
            plt.grid(True)
            plt.gca().get_xaxis().set_ticklabels([])
            plt.gca().get_yaxis().set_ticklabels([])

        plt.subplots_adjust(wspace=0, left=0.01, right=0.99)
        plt.savefig(vae_results / f"latent_space_T{time_step:02d}.png")
        plt.close()


def plot_tsne(z_loc, vae_results, wildfire_dataset: "WildFireDataset", max_samples=1000):
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)

    batch_size = z_loc.shape[0]
    logger.info(f"Size of the dataset: {batch_size}")
    diurnality = wildfire_dataset[:].diurnality
    if batch_size > max_samples:
        logger.info(f"Randomly subsample dataset to {max_samples} samples")
        batch_size = max_samples
        selected = np.arange(batch_size)
        np.random.shuffle(selected)
        selected = selected[:batch_size]
        selected.sort()
        z_loc = z_loc[selected, :, :]
        diurnality = diurnality[selected]
    logger.debug(f"z_loc shape {z_loc.shape}")
    logger.debug(f"diurnality shape {diurnality.shape}")

    z_loc = z_loc.reshape(-1, 10)
    logger.info(f"Fitting t-SNE model with z_loc (shape: {z_loc.shape})")
    z_embed = model_tsne.fit_transform(z_loc)
    x_lim = (z_embed[:, 0].min(), z_embed[:, 0].max())
    y_lim = (z_embed[:, 1].min(), z_embed[:, 1].max())

    z_embed = z_embed.reshape(batch_size, 7, 2).swapaxes(0, 1)
    logger.debug(f"z_embed shape {z_embed.shape}")
    plt.figure(figsize=(20, 4))

    logger.info(f"Plotting t-SNE")
    for i, zi in enumerate(z_embed):
        plt.subplot(1, 7, i + 1)
        logger.debug(f"zi shape {zi.shape}")

        plt.scatter(zi[diurnality == 1, 0], zi[diurnality == 1, 1], s=10, color=f"C0", label="day")
        plt.scatter(zi[diurnality == 0, 0], zi[diurnality == 0, 1], s=10, color=f"C1", label="night")
        plt.scatter(zi[diurnality == 1, 0][5], zi[diurnality == 1, 1][5], s=80, marker="s", color=f"red")
        plt.scatter(zi[diurnality == 0, 0][5], zi[diurnality == 0, 1][5], s=80, marker="s", color=f"blue")
        plt.legend(loc=1)
        plt.grid(True)
        plt.title(f"{mapping[i]} hr")
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.gca().get_xaxis().set_ticklabels([])
        plt.gca().get_yaxis().set_ticklabels([])

    plt.subplots_adjust(wspace=0, left=0.01, right=0.99)
    plt.savefig(vae_results / "tsne.png")
    plt.close()


def plot_epoch(vae_results, data, name, ylim=None):
    plt.figure()
    plt.plot(np.arange(len(data)) + 1, np.array(data))
    plt.xlim(0, len(data) + 1)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel(f"{name}")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.savefig(vae_results / f"{name.lower()}.png")
    plt.close()


def plot_forecast(vae: "VAE", vae_results, wildfire_dataset: "WildFireDataset", data_loader, max_samples=100):
    logger.info(f"Plotting forecasts")
    from matplotlib import rc
    from src.data.dataset import _ct
    batch_size = len(wildfire_dataset)
    rc('text', usetex=True)
    f_12, f_24 = None, None
    with torch.no_grad():
        for d in data_loader:
            if f_12 is None:
                f_12, f_24 = vae.forecast(_ct(d.diurnality), _ct(d.viirs[:, :5, :, :]))
            else:
                f_12_i, f_24_i = vae.forecast(_ct(d.diurnality), _ct(d.viirs[:, :5, :, :]))
                f_12 = np.concatenate((f_12, f_12_i), axis=0)
                f_24 = np.concatenate((f_24, f_24_i), axis=0)
    selected = slice(None)
    if batch_size > max_samples:
        logger.info(f"Randomly subsample dataset to {max_samples} samples")
        batch_size = max_samples
        selected = np.arange(batch_size)
        np.random.shuffle(selected)
        selected = selected[:batch_size]
        selected.sort()

    for i, idx in enumerate(wildfire_dataset[selected].index):
        plt.figure(figsize=(12, 6))
        for j in range(7):
            plt.subplot(2, 7, j + 1)
            plt.title(f'# {idx}')
            plt.imshow(wildfire_dataset[i].viirs[j, :, :])
            plt.axis('off')
            plt.title(f"{mapping[j]} hr" + r" $Y$")

        plt.subplot(2, 7, 13)
        plt.title(f'# {idx}')
        plt.imshow(f_12[i])
        plt.axis('off')
        plt.title(f"+12 hr" + r" $\hat{Y}$")

        plt.subplot(2, 7, 14)
        plt.title(f'# {idx}')
        plt.imshow(f_24[i])
        plt.axis('off')
        plt.title(f"+24 hr" + r" $\hat{Y}$")

        plt.subplots_adjust(wspace=0.01, left=0.01, right=0.99, hspace=0.01)
        plt.savefig(vae_results / f"forecast_{idx:05d}.png")
        plt.close()

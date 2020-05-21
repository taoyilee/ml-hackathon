# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from matplotlib import pyplot as plt

from src.data.wildfire import WildFireData
from src.vae import VAE

mapping = {0: -48, 1: -36, 2: -24, 3: -12, 4: 0, 5: 12, 6: 24}


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


def get_latent(vae: "VAE", wildfire_dataset: "WildFireData"):
    z_loc, z_scale = vae.encode(wildfire_dataset.diurnality, wildfire_dataset.viirs)
    return z_loc, z_scale


def plot_latent(vae: "VAE", vae_results, wildfire_dataset: "WildFireData"):
    z_loc, z_scale = get_latent(vae, wildfire_dataset)
    z_embed = z_loc[4].detach().cpu().numpy()
    x_lim = (z_embed[:, 0].min(), z_embed[:, 0].max())
    y_lim = (z_embed[:, 1].min(), z_embed[:, 1].max())
    plt.figure()
    diurnality = wildfire_dataset.diurnality.cpu().numpy()
    plt.scatter(z_embed[diurnality == 1, 0], z_embed[diurnality == 1, 1], s=20, color=f"C0", label="day")
    plt.scatter(z_embed[diurnality == 0, 0], z_embed[diurnality == 0, 1], s=20, color=f"C1", label="night")
    plt.legend(loc=1)
    plt.grid(True)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.tight_layout()
    plt.savefig(vae_results / "latent_space.png")
    plt.close()


def plot_tsne(vae: "VAE", vae_results, wildfire_dataset: "WildFireData"):
    z_loc, z_scale = get_latent(vae, wildfire_dataset)
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)

    z_loc = [z.detach().cpu().numpy() for z in z_loc]
    batch_size = z_loc[0].shape[0]
    z_loc = np.concatenate(z_loc)
    z_embed = model_tsne.fit_transform(z_loc)
    plt.figure(figsize=(20, 4))
    x_lim = (z_embed[:, 0].min(), z_embed[:, 0].max())
    y_lim = (z_embed[:, 1].min(), z_embed[:, 1].max())
    diurnality = wildfire_dataset.diurnality.cpu().numpy()

    for i in range(7):
        plt.subplot(1, 7, i + 1)
        zi = z_embed[batch_size * i:batch_size * (i + 1), :]
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


def plot_forecast(vae: "VAE", vae_results, wildfire_dataset: "WildFireData"):
    from matplotlib import rc
    rc('text', usetex=True)

    f_12, f_24 = vae.forecast(wildfire_dataset.diurnality, wildfire_dataset.viirs[:, :5, :, :])
    f_12 = f_12.cpu().detach().numpy().reshape(-1, 30, 30)
    f_24 = f_24.cpu().detach().numpy().reshape(-1, 30, 30)

    viirs = wildfire_dataset.viirs.cpu().numpy()
    for i, idx in enumerate(wildfire_dataset.indexes):
        plt.figure(figsize=(12, 6))
        for j in range(7):
            plt.subplot(2, 7, j + 1)
            plt.title(f'# {idx}')
            plt.imshow(viirs[i, j, :, :])
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

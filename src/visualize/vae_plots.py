# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from src.data.dataset import WildFireDataset


def plot_trajectory(csv_path, output_path):
    df = pd.read_csv(csv_path, index_col=0, dtype={"index": int})
    plt.figure(figsize=(14, 12), num=0)
    time_step = [-48, -36, -24, -12, 0, 12, 24]
    df = df.sample(frac=1)
    cmap = cm.get_cmap("coolwarm")
    x_lim = (df[['-48_h_0', '-36_h_0', '-24_h_0', '-12_h_0', '0_h_0', '12_h_0', '24_h_0']].min().min(),
             df[['-48_h_0', '-36_h_0', '-24_h_0', '-12_h_0', '0_h_0', '12_h_0', '24_h_0']].max().max())
    y_lim = (df[['-48_h_1', '-36_h_1', '-24_h_1', '-12_h_1', '0_h_1', '12_h_1', '24_h_1']].min().min(),
             df[['-48_h_1', '-36_h_1', '-24_h_1', '-12_h_1', '0_h_1', '12_h_1', '24_h_1']].max().max())

    df = df[:36]
    df = df.sort_values(by="index")
    wildfire = WildFireDataset(load_records=df["index"].to_numpy())
    idx = 0
    for _, row in df.iterrows():
        plt.figure(num=0)

        plt.subplot(6, 6, idx + 1)
        idx += 1
        for j, (t1, t2) in enumerate(zip(time_step[:-1], time_step[1:])):
            plt.arrow(row[f'{t1}_h_0'], row[f'{t1}_h_1'],
                      row[f'{t2}_h_0'] - row[f'{t1}_h_0'],
                      row[f'{t2}_h_1'] - row[f'{t1}_h_1']
                      , color=cmap(j / 6)[:3]
                      , width=0.3)
            # , linewidth=((j + 1) * 0.4) ** 2)
        # plt.hlines(-20, 0, x_lim[1])
        # plt.vlines(0, -20, y_lim[1])
        plt.text(x_lim[0], y_lim[0], int(row["index"]))
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.gca().get_xaxis().set_ticklabels([])
        plt.gca().get_yaxis().set_ticklabels([])

        plt.scatter(row[f'{time_step[0]}_h_0'], row[f'{time_step[0]}_h_1'], marker="^", s=40, color="C0")
        plt.scatter(row[f'{time_step[4]}_h_0'], row[f'{time_step[4]}_h_1'], marker="o", s=40, color="C1")
        plt.scatter(row[f'{time_step[-1]}_h_0'], row[f'{time_step[-1]}_h_1'], marker="s", s=40, color="C2")
        plt.grid(True)

        plt.figure(1)
        wildfire_record = wildfire.get_by_original_indexes(int(row["index"]))
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.title(f'{-12 * (4 - i)} hours')
            plt.imshow(wildfire_record[0][4 - i])
            plt.axis('off')

        # Plt Y detections
        for i in range(2):
            plt.subplot(2, 5, i + 5 + 1)
            plt.title(f'+{12 * (i + 1)} hours')
            plt.imshow(wildfire_record[1][i])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(Path(f"vae_results/details_{int(row['index'])}.png"))
        plt.close(1)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(output_path)


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


def plot_latent(vae, vae_results, wildfire_dataset):
    z_loc, z_scale = vae.encoder(wildfire_dataset.observed_0, wildfire_dataset.diurnality)
    z_embed = z_loc.detach().cpu().numpy()
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


def plot_tsne(vae, vae_results, wildfire_dataset):
    z_loc, z_scale = vae.encoder(wildfire_dataset.observed_0, wildfire_dataset.diurnality)
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
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
    plt.savefig(vae_results / "tsne.png")
    plt.close()


def plot_elbo(vae_results, elbo_loss_log):
    plt.figure()
    plt.plot(np.arange(len(elbo_loss_log)) + 1, -np.array(elbo_loss_log))
    plt.xlim(0, len(elbo_loss_log) + 1)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.ylim(-300, 0)
    plt.savefig(vae_results / "elbo.png")
    plt.close()

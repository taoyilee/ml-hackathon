# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, cm

from dataset import WildFireDataset


def plot_conditional_samples_ssvae(ssvae, visdom_session):
    """
    This is a method to do conditional sampling in visdom
    """
    vis = visdom_session
    ys = {}
    for i in range(10):
        ys[i] = torch.zeros(1, 10)
        ys[i][0, i] = 1
    xs = torch.zeros(1, 784)

    for i in range(10):
        images = []
        for rr in range(100):
            # get the loc from the model
            sample_loc_i = ssvae.model(xs, ys[i])
            img = sample_loc_i[0].view(1, 28, 28).cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)


def plot_llk_viirs(train_elbo):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy as sp
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    data = np.concatenate([np.arange(len(train_elbo))[:, sp.newaxis], -train_elbo[:, sp.newaxis]], axis=1)
    df = pd.DataFrame(data=data, columns=['Training Epoch', 'Training ELBO'])
    g = sns.FacetGrid(df, size=10, aspect=1.5)
    g.map(plt.scatter, "Training Epoch", "Training ELBO")
    g.map(plt.plot, "Training Epoch", "Training ELBO")

    output = Path('./vae_results/')
    output.mkdir(exist_ok=True)
    plt.savefig(output / "viirs_train_elbo_vae.png")
    plt.close('all')


def plot_llk(train_elbo, test_elbo):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy as sp
    import seaborn as sns
    plt.figure(figsize=(30, 10))
    sns.set_style("whitegrid")
    data = np.concatenate([np.arange(len(test_elbo))[:, sp.newaxis], -test_elbo[:, sp.newaxis]], axis=1)
    df = pd.DataFrame(data=data, columns=['Training Epoch', 'Test ELBO'])
    g = sns.FacetGrid(df, size=10, aspect=1.5)
    g.map(plt.scatter, "Training Epoch", "Test ELBO")
    g.map(plt.plot, "Training Epoch", "Test ELBO")

    output = Path('./vae_results/')
    output.mkdir(exist_ok=True)
    plt.savefig(output / "test_elbo_vae.png")
    plt.close('all')


def plot_vae_samples(vae, visdom_session):
    vis = visdom_session
    x = torch.zeros([1, 784])
    for i in range(10):
        images = []
        for rr in range(100):
            # get loc from the model
            sample_loc_i = vae.model(x)
            img = sample_loc_i[0].view(1, 28, 28).cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)


def mnist_test_tsne(vae=None, test_loader=None):
    """
    This is used to generate a t-sne embedding of the vae
    """
    name = 'VAE'
    data = test_loader.dataset.test_data.float()
    mnist_labels = test_loader.dataset.test_labels
    z_loc, z_scale = vae.encoder(data)
    plot_tsne(z_loc, mnist_labels, name)


def mnist_test_tsne_ssvae(name=None, ssvae=None, test_loader=None):
    """
    This is used to generate a t-sne embedding of the ss-vae
    """
    if name is None:
        name = 'SS-VAE'
    data = test_loader.dataset.test_data.float()
    mnist_labels = test_loader.dataset.test_labels
    z_loc, z_scale = ssvae.encoder_z([data, mnist_labels])
    plot_tsne(z_loc, mnist_labels, name)


def plot_tsne(z_loc, classes, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    output = Path('./vae_results/')
    output.mkdir(exist_ok=True)
    for ic in range(10):
        ind_vec = np.zeros_like(classes)
        ind_vec[:, ic] = 1
        ind_class = classes[:, ic] == 1
        color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig(output / f"{name}_embedding_{ic}.png")
    fig.savefig(output / f"{name}_embedding.png")


def viirs_tsne(vae=None, data_loader=None):
    """
    This is used to generate a t-sne embedding of the vae
    """
    name = 'VAE'
    z_loc, z_scale = vae.encoder(torch.tensor(data_loader.dataset.x).cuda())
    plot_viirs_tsne(z_loc, data_loader.dataset.timestep, data_loader.dataset.indexes, name)


def plot_viirs_tsne(z_loc, timestep, index, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    output = Path('./vae_results/')
    output.mkdir(exist_ok=True)
    timestep_mapping = {-4: -48, -3: -36, -2: -24, -1: -12, 0: 0, 1: 12, 2: 24}
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)

    df = pd.DataFrame(columns=["index", "-48_h_0", "-48_h_1"])
    df["index"] = np.unique(index)

    x_lim = (z_embed[:, 0].min(), z_embed[:, 0].max())
    y_lim = (z_embed[:, 1].min(), z_embed[:, 1].max())
    fig = plt.figure(figsize=(16, 16 / 7))
    for i, t in enumerate([-4, -3, -2, -1, 0, 1, 2]):
        plt.subplot(1, 7, i + 1)
        plt.title(f"{timestep_mapping[t]} hr")
        df.loc[df["index"] == index[timestep == t], f"{timestep_mapping[t]}_h_0"] = z_embed[timestep == t, 0]
        df.loc[df["index"] == index[timestep == t], f"{timestep_mapping[t]}_h_1"] = z_embed[timestep == t, 1]
        plt.scatter(z_embed[timestep == t, 0], z_embed[timestep == t, 1], s=5, label=timestep_mapping[t],
                    color=f"C{i}")
        plt.grid(True)

        if i != 0:
            plt.gca().get_yaxis().set_ticklabels([])
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    plt.subplots_adjust(wspace=0.06, left=0.04, right=0.99)
    fig.savefig(output / f"{name}_viirs_embedding.png")
    plt.close()
    print(df)
    df.to_csv(output / f"{name}_viirs_embedding.csv")


def plot_trajectory(csv_path, output_path):
    df = pd.read_csv(csv_path, index_col=0, dtype={"index": int})
    print(df.keys(), df.dtypes)
    figure = plt.figure(figsize=(14, 12), num=0)
    time_step = [-48, -36, -24, -12, 0, 12, 24]
    df = df.sample(frac=1)
    cmap = cm.get_cmap("coolwarm")
    x_lim = (df[['-48_h_0', '-36_h_0', '-24_h_0', '-12_h_0', '0_h_0', '12_h_0', '24_h_0']].min().min(),
             df[['-48_h_0', '-36_h_0', '-24_h_0', '-12_h_0', '0_h_0', '12_h_0', '24_h_0']].max().max())
    y_lim = (df[['-48_h_1', '-36_h_1', '-24_h_1', '-12_h_1', '0_h_1', '12_h_1', '24_h_1']].min().min(),
             df[['-48_h_1', '-36_h_1', '-24_h_1', '-12_h_1', '0_h_1', '12_h_1', '24_h_1']].max().max())
    print(x_lim, y_lim)
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

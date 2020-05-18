# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pandas as pd
import torch


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
    plt.figure(figsize=(30, 10))
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

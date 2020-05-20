import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO

from my_vae import VAE
from src.utils import beta_to_mean_std

if __name__ == "__main__":
    import h5py
    import pandas as pd

    n_steps = 2000
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    n = 500
    head = 0.8
    heads = int(n * 0.8)

    with h5py.File("data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train.hdf5", 'r') as h5_ptr:
        indexes = np.arange(len(h5_ptr['datetime']))
        np.random.shuffle(indexes)
        indexes = indexes[:n]
        indexes.sort()
        data = {k: h5_ptr[k][indexes] for k in h5_ptr.keys() if k in ['datetime', 'observed']}
        for k, v in data.items():
            print(f"{k}: {v.shape}")
    data["diurnality"] = pd.to_datetime(data["datetime"])
    data["diurnality"] = data["diurnality"].hour
    print(data["diurnality"].min(), data["diurnality"].max())
    data["diurnality"] = (np.bitwise_and(data["diurnality"] > 6, data["diurnality"] <= 18)).astype(np.float32)
    data["observed"] = data["observed"].astype(np.float32)
    unique, count = np.unique(data["diurnality"], return_counts=True)
    count = count / np.sum(count)
    print({u: c for u, c in zip(unique, count)})

    diurnality = torch.tensor(data["diurnality"]).cuda()
    observed_0 = torch.tensor(data["observed"][:, 0, :, :]).cuda()
    print(diurnality.shape, diurnality.min(), diurnality.max())
    print(observed_0.shape, observed_0.min(), observed_0.max())

    vae = VAE()

    svi = SVI(vae.model, vae.guide, vae.optimizer, loss=Trace_ELBO())
    elbo_loss_log = []
    for step in range(n_steps):
        elbo_loss_log.append(svi.step(diurnality, observed_0) / n)
        if (step + 1) % 200 == 0:
            print(f"[Iter {step + 1}] ELBO: {elbo_loss_log[-1] :.3f}")

    alpha_q = pyro.param("alpha_q").item()
    beta_q = pyro.param("beta_q").item()

    inferred_mean, inferred_std = beta_to_mean_std(alpha_q, beta_q)

    print(f"{alpha_q:.2f} {beta_q:.2f} diurnality estimate {inferred_mean:.2f}+/-{inferred_std:.2f}")
    plt.figure()
    plt.plot(np.arange(len(elbo_loss_log)) + 1, -np.array(elbo_loss_log))
    plt.xlim(0, len(elbo_loss_log) + 1)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.savefig("/home/tylee/PycharmProjects/wildfire/vae_results/elbo.png")
    plt.close()

    # T-SNE
    z_loc, z_scale = vae.encoder(observed_0, diurnality)
    from sklearn.manifold import TSNE

    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)

    x_lim = (z_embed[:, 0].min(), z_embed[:, 0].max())
    y_lim = (z_embed[:, 1].min(), z_embed[:, 1].max())
    plt.figure()

    diurnality = diurnality.cpu().numpy()
    plt.scatter(z_embed[diurnality == 1, 0], z_embed[diurnality == 1, 1], s=20, color=f"C0", label="day")
    plt.scatter(z_embed[diurnality == 0, 0], z_embed[diurnality == 0, 1], s=20, color=f"C1", label="night")
    plt.legend(loc=1)
    plt.grid(True)
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.tight_layout()
    plt.savefig("/home/tylee/PycharmProjects/wildfire/vae_results/tsne.png")
    plt.close()

    observed_0 = observed_0.cpu().numpy()
    counter = 0
    plt.figure(figsize=(10, 10))
    fn = 1
    for i, (idx, obs) in enumerate(zip(indexes[diurnality == 1], observed_0[diurnality == 1])):
        plt.subplot(6, 6, counter + 1)
        plt.title(f'# {idx}')
        plt.imshow(observed_0[i])
        plt.axis('off')

        counter += 1
        if counter == 36:
            plt.tight_layout()
            plt.savefig(f"/home/tylee/PycharmProjects/wildfire/vae_results/viirs_day_{fn:05d}.png")
            plt.close()
            fn += 1
            counter = 0
            plt.figure(figsize=(10, 10))

    plt.figure(figsize=(10, 10))
    fn = 1
    for i, (idx, obs) in enumerate(zip(indexes[diurnality == 0], observed_0[diurnality == 0])):
        plt.subplot(6, 6, counter + 1)
        plt.title(f'# {idx}')
        plt.imshow(observed_0[i])
        plt.axis('off')

        counter += 1
        if counter == 36:
            plt.tight_layout()
            plt.savefig(f"/home/tylee/PycharmProjects/wildfire/vae_results/viirs_night_{fn:05d}.png")
            plt.close()
            fn += 1
            counter = 0
            plt.figure(figsize=(10, 10))

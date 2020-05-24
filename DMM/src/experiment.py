import configparser as cp
import json
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from sacred import Experiment
from sacred import SETTINGS
from sacred.run import Run
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from src import logger
from src.data import WildFireDataset
from src.eval import eval_light
from src.utils import beta_to_mean_std
from src.vae import VAE, VAEConfig

SETTINGS['CAPTURE_MODE'] = 'no'
ex = Experiment('deep_markov_model')


@ex.config
def cfg():
    config_file = "config.ini"
    config = cp.ConfigParser()
    config.read(config_file)

    temp = Path(config["DEFAULT"]["temp"])
    temp.mkdir(exist_ok=True)

    # noinspection PyUnusedLocal
    epochs = config["vae-training"].getint("epochs")
    # noinspection PyUnusedLocal
    batch_size = config["vae-training"].getint("batch_size")
    # noinspection PyUnusedLocal
    max_batch_steps = config["vae-training"].getint("max_batch_steps")
    # noinspection PyUnusedLocal
    z_dim = config["vae-training"].getint("z_dim")
    # noinspection PyUnusedLocal
    min_af = config["vae-training"].getfloat("min_af")
    # noinspection PyUnusedLocal
    annealing_epochs = config["vae-training"].getint("annealing_epochs")

    # noinspection PyUnusedLocal
    dropout_rate = config["vae-training"].getfloat("dropout_rate")
    # noinspection PyUnusedLocal
    rnn_dim = config["vae-training"].getint("rnn_dim")
    # noinspection PyUnusedLocal
    rnn_layers = config["vae-training"].getint("rnn_layers")
    # noinspection PyUnusedLocal
    num_iafs = config["vae-training"].getint("num_iafs")
    # noinspection PyUnusedLocal
    iaf_dim = config["vae-training"].getint("iaf_dim")
    # noinspection PyUnusedLocal
    transition_dim = config["vae-training"].getint("transition_dim")
    # noinspection PyUnusedLocal
    crnn_channel = config["vae-training"].getint("crnn_channel")
    # noinspection PyUnusedLocal
    emitter_channel = config["vae-training"].getint("emitter_channel")
    # noinspection PyUnusedLocal
    init_lr = config["vae-training"].getfloat("init_lr")
    # noinspection PyUnusedLocal
    eval_freq = config["vae-training"].getint("eval_freq")

    # noinspection PyUnusedLocal
    loader_workers = config["vae-training"].getint("loader_workers")


@ex.capture
def seed_random(_seed):
    pyro.set_rng_seed(_seed)
    torch.manual_seed(_seed)
    # TODO: making cuDNN deterministic seems to have no effect ?? Dropout layer will be different each time
    torch.cuda.manual_seed_all(_seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(_seed)


@ex.capture
def get_vae_config(z_dim, dropout_rate, rnn_dim, rnn_layers, num_iafs, iaf_dim, transition_dim,
                   crnn_channel, emitter_channel, init_lr, _run: "Run"):
    return VAEConfig(z_dim=z_dim, image_dim=(30, 30), dropout_rate=dropout_rate, rnn_dim=rnn_dim,
                     rnn_layers=rnn_layers, num_iafs=num_iafs, iaf_dim=iaf_dim, transition_dim=transition_dim,
                     crnn_channel=crnn_channel, emitter_channel=emitter_channel, init_lr=init_lr)


@ex.main
def run(batch_size, max_batch_steps, epochs, annealing_epochs, temp, min_af, loader_workers, eval_freq, _run: "Run"):
    pyro.clear_param_store()
    _run.add_artifact(_run.config["config_file"])

    # Seed randomness for repeatability
    seed_random()

    # dataset
    wildfire_dataset = WildFireDataset(train=True, config_file="config.ini")
    data_loader = DataLoader(wildfire_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers)
    expected_batch_size = np.ceil(len(wildfire_dataset) / batch_size)
    expected_batch_size = max_batch_steps if max_batch_steps > 0 else expected_batch_size
    vae_config = get_vae_config()

    with open(temp / "vae_config.json", "w") as fptr:
        json.dump(vae_config.__dict__, fptr, indent=1)

    _run.add_artifact(temp / "vae_config.json")

    vae = VAE(vae_config)
    svi = SVI(vae.model, vae.guide, vae.optimizer, loss=Trace_ELBO())

    from src.data.dataset import _ct

    for step in trange(epochs, desc="Epoch: ", ascii=False, dynamic_ncols=True,
                       bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:40}{r_bar}'):
        if step < annealing_epochs:
            annealing_factor = min_af + (1.0 - min_af) * step / annealing_epochs
        else:
            annealing_factor = 1.0
        _run.log_scalar("annealing_factor", annealing_factor, step=step)

        epoch_elbo = 0.0
        epoch_time_slices = 0
        for batch_steps_i, d in tqdm(enumerate(data_loader), desc="Batch: ", ascii=False, dynamic_ncols=True,
                                     bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:40}{r_bar}',
                                     total=expected_batch_size,
                                     leave=False):
            epoch_elbo += svi.step(_ct(d.diurnality), _ct(d.viirs), _ct(d.land_cover), _ct(d.latitude),
                                   _ct(d.longitude), _ct(d.meteorology), annealing_factor)
            epoch_time_slices += d.viirs.shape[0] * d.viirs.shape[0]
            if 0 < max_batch_steps == batch_steps_i:
                break
        elbo = -epoch_elbo / epoch_time_slices
        print(f" [{step:05d}] ELBO: {elbo:.3f}", end="")
        _run.log_scalar("elbo", elbo, step=step)
        alpha = pyro.param("alpha").item()
        beta = pyro.param("beta").item()
        _run.log_scalar("alpha", alpha, step=step)
        _run.log_scalar("beta", beta, step=step)
        inferred_mean, inferred_std = beta_to_mean_std(alpha, beta)
        _run.log_scalar("inferred_mean", inferred_mean, step=step)
        _run.log_scalar("inferred_std", inferred_std, step=step)

        if eval_freq > 0 and step > 0 and step % eval_freq == 0:
            logger.info("Evaluating")
            eval_light(Path(_run.observers[0].dir), vae, data_loader, wildfire_dataset, step)
            vae.train()

    torch.save(vae.state_dict(), temp / "model_final.pt")
    _run.add_artifact(temp / "model_final.pt")
    vae.optimizer.save(temp / "optimizer.pt")
    _run.add_artifact(temp / "optimizer.pt")

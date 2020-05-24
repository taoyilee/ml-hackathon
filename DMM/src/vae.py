import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.optim import ClippedAdam
from torch.distributions import constraints as constraints

from src.decoder import Decoder
from src.encoder import GatedTransition, Combiner, ConvRNN


class VAEConfig:
    def __init__(self, z_dim=10, image_dim=(30, 30), dropout_rate=0.2, rnn_dim=200, rnn_layers=1, num_iafs=0,
                 iaf_dim=50, transition_dim=200, crnn_channel=16, emitter_channel=16, init_lr=1e-3, use_lstm=False):
        self.z_dim = z_dim
        self.image_dim = image_dim
        self.dropout_rate = dropout_rate
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        self.num_iafs = num_iafs
        self.iaf_dim = iaf_dim
        self.transition_dim = transition_dim
        self.crnn_channel = crnn_channel
        self.emitter_channel = emitter_channel

        self.init_lr = init_lr
        self.use_lstm = use_lstm


class VAE(nn.Module):

    def __init__(self, _c: "VAEConfig"):
        super().__init__()
        self._c = _c
        self.image_flatten_dim = _c.image_dim[0] * _c.image_dim[1]

        adam_params = {"lr": _c.init_lr, "betas": (0.96, 0.999), "clip_norm": 10.0, "lrd": 0.99996, "weight_decay": 2.0}
        self.optimizer = ClippedAdam(adam_params)

        self.emitter = Decoder(_c.z_dim, _c.emitter_channel, dropout_p=_c.dropout_rate)
        self.trans = GatedTransition(_c.z_dim, _c.transition_dim)
        self.combiner = Combiner(_c.z_dim, _c.rnn_dim)

        self.crnn = ConvRNN(_c.image_dim, _c.rnn_dim, _c.rnn_layers, _c.dropout_rate,
                            use_lstm=_c.use_lstm, channels=_c.crnn_channel)
        self.iafs = [affine_autoregressive(_c.z_dim, hidden_dims=[_c.iaf_dim]) for _ in range(_c.num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        self.z_0 = nn.Parameter(torch.zeros(_c.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(_c.z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, _c.rnn_dim))
        if _c.use_lstm:
            self.c_0 = nn.Parameter(torch.zeros(1, 1, _c.rnn_dim))
        self.cuda()

    def model(self, diurnality, viirs_observed, land_cover, latitude, longitude, meteorology, annealing_factor=1.0):
        # land_cover.shape: [128, 17, 30, 30]
        pyro.module("vae", self)
        batch_size = diurnality.shape[0]
        T_max = viirs_observed.size(1)
        z_prev = self.z_0.expand(batch_size, self.z_0.size(0))
        alpha0 = torch.tensor(10.0, device=diurnality.device)
        beta0 = torch.tensor(10.0, device=diurnality.device)
        diurnal_ratio = pyro.sample("diurnal_ratio", dist.Beta(alpha0, beta0))

        with pyro.plate("data", batch_size):
            diurnal_ = pyro.sample("diurnal_", dist.Bernoulli(diurnal_ratio), obs=diurnality).long()
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc, z_scale = self.trans(z_prev, land_cover)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc[diurnal_, :], z_scale[diurnal_, :]).to_event(1))
                image_p = self.emitter(z_t)
                image = pyro.sample(f"image_{t}", dist.Bernoulli(image_p).to_event(1),
                                    obs=viirs_observed[:, t - 1, :, :].reshape(-1, self.image_flatten_dim))
                z_prev = z_t
            return image_p, image

    def guide(self, diurnality, viirs_observed, land_cover, latitude, longitude, meteorology, annealing_factor=1.0):
        T_max = viirs_observed.size(1)
        batch_size = diurnality.shape[0]
        pyro.module("vae", self)
        c_0_contig, h_0_contig = self.rnn_state_contig(batch_size)

        rnn_output = self.crnn(viirs_observed, h_0_contig, c_0_contig)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))
        _constraint = constraints.interval(1, 100)
        alpha_q = pyro.param("alpha", torch.tensor(10.0, device=diurnality.device), constraint=_constraint)
        beta_q = pyro.param("beta", torch.tensor(10.0, device=diurnality.device), constraint=_constraint)
        pyro.sample("diurnal_ratio", dist.Beta(alpha_q, beta_q))
        with pyro.plate("data", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc_q, z_scale_q = self.combiner(z_prev, rnn_output[:, t - 1, :], diurnality)
                z_t = self.sample_latent_space(annealing_factor, batch_size, t, z_loc_q, z_scale_q)
                z_prev = z_t

    def rnn_state_contig(self, batch_size):
        h_0_contig = self.h_0.expand(1, batch_size, self.crnn.hidden_size).contiguous()
        c_0_contig = self.c_0.expand(1, batch_size, self.crnn.hidden_size).contiguous() if self._c.use_lstm else None
        return c_0_contig, h_0_contig

    def sample_latent_space(self, annealing_factor, batch_size, t, z_loc_q, z_scale_q):
        if len(self.iafs) > 0:
            z_dist = TransformedDistribution(dist.Normal(z_loc_q, z_scale_q), self.iafs)
            assert z_dist.event_shape == (self.z_q_0.size(0),)
            assert z_dist.batch_shape[-1:] == (batch_size,)
        else:
            z_dist = dist.Normal(z_loc_q, z_scale_q)
            assert z_dist.event_shape == ()
            assert z_dist.batch_shape[-2:] == (batch_size, self.z_q_0.size(0))
        with pyro.poutine.scale(scale=annealing_factor):
            if len(self.iafs) > 0:
                # in output of normalizing flow, all dimensions are correlated (event shape is not empty)
                z_t = pyro.sample(f"z_{t}", z_dist)
            else:
                # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                z_t = pyro.sample(f"z_{t}", z_dist.to_event(1))
        return z_t

    def encode(self, diurnality, viirs_observed):
        self.eval()

        T_max = viirs_observed.size(1)
        batch_size = diurnality.shape[0]

        c_0_contig, h_0_contig = self.rnn_state_contig(batch_size)

        rnn_output = self.crnn(viirs_observed, h_0_contig, c_0_contig)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))

        z_loc, z_scale = [], []
        with pyro.plate("data", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc_q, z_scale_q = self.combiner(z_prev, rnn_output[:, t - 1, :], diurnality)
                z_loc.append(z_loc_q)
                z_scale.append(z_scale_q)
                z_prev = self.sample_latent_space(1.0, batch_size, t, z_loc_q, z_scale_q)

        return np.array([z.cpu().numpy() for z in z_loc]), np.array([z.cpu().numpy() for z in z_scale])

    def forecast(self, diurnality, viirs_observed, land_cover, latitude, longitude, meteorology):
        assert viirs_observed.shape[1:] == (5, 30, 30), \
            f"viirs_observed.shape[1:] = {viirs_observed.shape[1:]} != (5, 30, 30)"
        T_max = viirs_observed.size(1)
        batch_size = diurnality.shape[0]
        diurnality = diurnality.long()
        self.eval()

        c_0_contig, h_0_contig = self.rnn_state_contig(batch_size)
        rnn_output = self.crnn(viirs_observed, h_0_contig, c_0_contig)
        z_prev = self.z_q_0.expand(viirs_observed.size(0), self.z_q_0.size(0))

        with pyro.plate("data", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc_q, z_scale_q = self.combiner(z_prev, rnn_output[:, t - 1, :], diurnality)
                z_prev = self.sample_latent_space(1.0, batch_size, t, z_loc_q, z_scale_q)

        forecast_12, z_prev = self.forecast_next_step(diurnality, land_cover, latitude, longitude, meteorology, z_prev)
        forecast_24, _ = self.forecast_next_step(diurnality, land_cover, latitude, longitude, meteorology, z_prev)

        return forecast_12, forecast_24

    def forecast_next_step(self, diurnality, land_cover, latitude, longitude, meteorology, z_prev):
        z_loc, z_scale = self.trans(z_prev, land_cover)
        z_prev = dist.Normal(z_loc[diurnality, :], z_scale[diurnality, :])()
        forecast = self.emitter(z_prev).cpu().detach().reshape(-1, 30, 30)
        return forecast, z_prev

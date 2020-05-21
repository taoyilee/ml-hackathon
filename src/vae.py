import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.distributions.transforms import affine_autoregressive
from pyro.optim import ClippedAdam
from torch.distributions import constraints as constraints

from src.decoder import Decoder
from src.encoder import GatedTransition, Combiner


class VAE(nn.Module):
    adam_params = {"lr": 1e-3, "betas": (0.96, 0.999), "clip_norm": 10, "lrd": 0.99996, "weight_decay": 2.0}

    def __init__(self, z_dim=5, image_dim=(30, 30), dropout_rate=0.25, rnn_dim=600, num_layers=1, num_iafs=2,
                 iaf_dim=50, transition_dim=200):
        super().__init__()
        self.image_flatten_dim = image_dim[0] * image_dim[1]
        self.optimizer = ClippedAdam(self.adam_params)

        self.z_dim = z_dim
        self.hidden_dim = 400
        self.channel = 8
        self.dropout_p = dropout_rate
        self.emitter = Decoder(self.z_dim, self.channel, dropout_p=self.dropout_p)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)

        rnn_dropout_rate = 0. if num_layers == 1 else dropout_rate
        input_dim = image_dim[0] * image_dim[1]
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=num_layers,
                          dropout=rnn_dropout_rate)

        self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))

        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.cuda()

    def model(self, diurnality, viirs_observed, annealing_factor=1.0):
        pyro.module("vae", self)

        batch_size = diurnality.shape[0]

        T_max = viirs_observed.size(1)

        z_prev = self.z_0.expand(viirs_observed.size(0), self.z_0.size(0))

        alpha0 = torch.tensor(10.0, device=diurnality.device)
        beta0 = torch.tensor(10.0, device=diurnality.device)
        diurnal_ratio = pyro.sample("diurnal_ratio", dist.Beta(alpha0, beta0))

        with pyro.plate("data", batch_size):
            diurnal_ = pyro.sample("diurnal_", dist.Bernoulli(diurnal_ratio), obs=diurnality).long()
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc, z_scale = self.trans(z_prev)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc[diurnal_, :], z_scale[diurnal_, :]).to_event(1))
                image_p = self.emitter(z_t)
                image = pyro.sample(f"image_{t}", dist.Bernoulli(image_p).to_event(1),
                                    obs=viirs_observed[:, t - 1, :, :].reshape(-1, self.image_flatten_dim))
                z_prev = z_t
            return image_p, image

    def guide(self, diurnality, viirs_observed, annealing_factor=1.0):
        T_max = viirs_observed.size(1)
        batch_size = diurnality.shape[0]
        pyro.module("vae", self)

        h_0_contig = self.h_0.expand(1, viirs_observed.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(torch.flip(viirs_observed.view(batch_size, -1, 900), dims=[1]), h_0_contig)
        rnn_output = torch.flip(rnn_output, dims=[1])
        z_prev = self.z_q_0.expand(viirs_observed.size(0), self.z_q_0.size(0))

        alpha_q = pyro.param("alpha", torch.tensor(10.0, device=diurnality.device),
                             constraint=constraints.interval(1, 100))
        beta_q = pyro.param("beta", torch.tensor(10.0, device=diurnality.device),
                            constraint=constraints.interval(1, 100))
        diurnal_ratio_q = pyro.sample("diurnal_ratio", dist.Beta(alpha_q, beta_q))

        with pyro.plate("data", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc_q, z_scale_q = self.combiner(z_prev, rnn_output[:, t - 1, :], diurnality)
                z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc_q, z_scale_q).to_event(1))
                z_prev = z_t

    def encode(self, diurnality, viirs_observed):
        T_max = viirs_observed.size(1)
        batch_size = diurnality.shape[0]
        pyro.module("vae", self)
        self.rnn.eval()

        h_0_contig = self.h_0.expand(1, viirs_observed.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(torch.flip(viirs_observed.view(batch_size, -1, 900), dims=[1]), h_0_contig)
        rnn_output = torch.flip(rnn_output, dims=[1])
        z_prev = self.z_q_0.expand(viirs_observed.size(0), self.z_q_0.size(0))

        z_loc = []
        z_scale = []
        with pyro.plate("data", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc_q, z_scale_q = self.combiner(z_prev, rnn_output[:, t - 1, :], diurnality)
                z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc_q, z_scale_q).to_event(1))
                z_prev = z_t
                z_loc.append(z_loc_q)
                z_scale.append(z_scale_q)
        return z_loc, z_scale

    def forecast(self, diurnality, viirs_observed):
        assert viirs_observed.shape[1:] == (5, 30, 30), \
            f"viirs_observed.shape[1:] = {viirs_observed.shape[1:]} != (5, 30, 30)"

        T_max = viirs_observed.size(1)
        batch_size = diurnality.shape[0]
        diurnality = diurnality.long()

        pyro.module("vae", self)
        self.rnn.eval()
        self.combiner.eval()
        self.trans.eval()

        h_0_contig = self.h_0.expand(1, viirs_observed.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(torch.flip(viirs_observed.view(batch_size, -1, 900), dims=[1]), h_0_contig)
        rnn_output = torch.flip(rnn_output, dims=[1])
        z_prev = self.z_q_0.expand(viirs_observed.size(0), self.z_q_0.size(0))

        with pyro.plate("data", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc_q, z_scale_q = self.combiner(z_prev, rnn_output[:, t - 1, :], diurnality)
                z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc_q, z_scale_q).to_event(1))
                z_prev = z_t

        z_loc, z_scale = self.trans(z_prev)
        z_prev = dist.Normal(z_loc[diurnality, :], z_scale[diurnality, :])()
        forecast_12 = self.emitter(z_prev)

        z_loc, z_scale = self.trans(z_prev)
        z_prev = dist.Normal(z_loc[diurnality, :], z_scale[diurnality, :])()
        forecast_24 = self.emitter(z_prev)

        return forecast_12, forecast_24

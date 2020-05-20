import pyro
import torch
from pyro import distributions as dist
from pyro.optim import Adam
from torch import nn
from torch.distributions import constraints as constraints

from src.decoder import Decoder
from src.encoder import Encoder


class VAE(nn.Module):
    def __init__(self, z_dim=2, image_dim=(30, 30)):
        super().__init__()
        self.image_flatten_dim = image_dim[0] * image_dim[1]
        adam_params = {"lr": 1e-3, "betas": (0.9, 0.999)}
        # self.optimizer = Adam(adam_params, clip_args={"clip_norm": 100.0})
        self.optimizer = Adam(adam_params)
        self.z_dim = z_dim
        self.hidden_dim = 400
        self.encoder = Encoder(self.z_dim, self.hidden_dim)
        self.decoder = Decoder(self.z_dim, self.hidden_dim)
        self.cuda()

    def model(self, diurnality, observed_0):
        batch_size = diurnality.shape[0]
        alpha0 = torch.tensor(10.0, device=diurnality.device)
        beta0 = torch.tensor(10.0, device=diurnality.device)
        z_loc = torch.zeros((2, self.z_dim), device=diurnality.device)
        z_scale = torch.ones((2, self.z_dim), device=diurnality.device)

        diurnal_ratio = pyro.sample("diurnal_ratio", dist.Beta(alpha0, beta0))
        with pyro.plate("data", batch_size):
            pyro.module("decoder", self.decoder)
            diurnal_ = pyro.sample("diurnal_", dist.Bernoulli(diurnal_ratio), obs=diurnality).long()
            z_dist = dist.Normal(z_loc[diurnal_, :], z_scale[diurnal_, :]).to_event(1)
            z = pyro.sample("latent", z_dist)
            image = self.decoder(z)
            pyro.sample("image_binarized", dist.Bernoulli(image).to_event(1),
                        obs=observed_0.reshape(-1, self.image_flatten_dim))
            return image

    def guide(self, diurnality, observed_0):
        pyro.module("encoder", self.encoder)

        alpha_q = pyro.param("alpha_q", torch.tensor(10.0, device=diurnality.device),
                             constraint=constraints.interval(1, 100))
        beta_q = pyro.param("beta_q", torch.tensor(10.0, device=diurnality.device),
                            constraint=constraints.interval(1, 100))
        z_loc_q, z_scale_q = self.encoder(observed_0, diurnality)
        pyro.sample("diurnal_ratio", dist.Beta(alpha_q, beta_q))
        pyro.sample("latent", dist.Normal(z_loc_q, z_scale_q).to_event(1))

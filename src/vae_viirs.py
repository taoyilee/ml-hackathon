import pyro
import pyro.contrib.examples.util  # patches torchvision
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
import torch.nn as nn


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, image_dimension=(30, 30)):
        super().__init__()
        # setup the three linear transformations used
        self.z_dim = z_dim
        self.image_flatten_dim = image_dimension[0] * image_dimension[1]
        self.fc1 = nn.Linear(self.image_flatten_dim + 1, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x, diurnal_assignment_q):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = torch.cat((x.reshape(-1, self.image_flatten_dim), diurnal_assignment_q.unsqueeze(-1)), dim=1)

        # if torch.sum(x) == 0:
        #     return torch.zeros(self.z_dim), 1 * torch.ones(self.z_dim)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim

        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        if torch.any(torch.isnan(z_loc)):
            raise ValueError(f"z_loc has nan")
        if torch.any(torch.isnan(z_scale)):
            raise ValueError(f"z_scale has nan")
        # print((torch.sum(x, dim=1) > 0).unsqueeze(-1).shape, z_loc.shape)
        z_loc = torch.where((torch.sum(x, dim=1) > 0).unsqueeze(-1), z_loc, torch.zeros_like(z_loc))
        # z_scale = torch.where((torch.sum(x, dim=1) > 0).unsqueeze(-1), z_scale, 1e-3 * torch.ones_like(z_scale))
        return z_loc, z_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, image_dimension=(30, 30)):
        super().__init__()
        # setup the two linear transformations used
        self.image_flatten_dim = image_dimension[0] * image_dimension[1]
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, self.image_flatten_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = torch.sigmoid(self.fc21(hidden))
        return loc_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=400, image_dimension=(30, 30), use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.image_flatten_dim = image_dimension[0] * image_dimension[1]
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x, diurnal):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        # setup hyperparameters for prior p(d)
        alpha0 = torch.tensor(1.0, device=x.device)
        beta0 = torch.tensor(1.0, device=x.device)
        diurnal_prob = pyro.sample("diurnal_prob", dist.Beta(alpha0, beta0))
        # setup hyperparameters for prior p(z)
        z_loc = torch.zeros(2, self.z_dim, dtype=x.dtype, device=x.device)
        z_scale = torch.ones(2, self.z_dim, dtype=x.dtype, device=x.device)

        with pyro.plate("data", x.shape[0]):
            # diurnal_assignment = pyro.sample('diurnal_assignment', dist.Bernoulli(diurnal_prior), obs=diurnal)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            diurnal_assign = pyro.sample("diurnal_assign", dist.Bernoulli(diurnal_prob), obs=diurnal)
            diurnal_assign = diurnal_assign.long()
            # print(diurnal_assign.shape, z_loc.shape)
            # print(z_loc[diurnal_assign, :].shape)
            z = pyro.sample("latent", dist.Normal(z_loc[diurnal_assign, :], z_scale[diurnal_assign, :]).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, self.image_flatten_dim))
            # return the loc so we can visualize it later
            # return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, diurnal):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        alpha_q = pyro.param("alpha_q", torch.tensor(1.0, device=x.device), constraint=constraints.positive)
        beta_q = pyro.param("beta_q", torch.tensor(1.0, device=x.device), constraint=constraints.positive)
        diurnal_prob = pyro.sample("diurnal_prob", dist.Beta(alpha_q, beta_q))
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            diurnal_assign_q = pyro.sample("diurnal_assign", dist.Bernoulli(diurnal_prob))
            z_loc, z_scale = self.encoder.forward(x, diurnal_assign_q)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

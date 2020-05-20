import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, image_dimension=(30, 30)):
        super().__init__()

        self.z_dim = z_dim
        self.image_flatten_dim = image_dimension[0] * image_dimension[1]
        self.fc1_day = nn.Linear(self.image_flatten_dim, hidden_dim)
        self.fc21_day = nn.Linear(hidden_dim, z_dim)
        self.fc22_day = nn.Linear(hidden_dim, z_dim)

        self.fc1_night = nn.Linear(self.image_flatten_dim, hidden_dim)
        self.fc21_night = nn.Linear(hidden_dim, z_dim)
        self.fc22_night = nn.Linear(hidden_dim, z_dim)

        self.softplus = nn.Softplus()

    def forward(self, observed_0, diurnality):
        observed_0 = observed_0.reshape(-1, self.image_flatten_dim)
        hidden = self.softplus(self.fc1_day(observed_0))
        condition = (diurnality == 1).unsqueeze(-1)
        z_loc = torch.where(condition, self.fc21_day(hidden), self.fc21_night(hidden))
        z_scale = torch.exp(torch.where(condition, self.fc22_day(hidden), self.fc22_night(hidden)))
        return z_loc, z_scale
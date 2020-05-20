import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, image_dimension=(30, 30)):
        super().__init__()

        self.image_flatten_dim = image_dimension[0] * image_dimension[1]
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, self.image_flatten_dim)

        self.softplus = nn.Softplus()

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        loc_img = torch.sigmoid(self.fc21(hidden))
        return loc_img

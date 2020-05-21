import torch
import torch.nn.functional as F
from torch import nn as nn


class Encoder(nn.Module):
    def __init__(self, z_dim, channels=4, hidden_dim=300, dropout_p=0.2):
        super().__init__()

        self.z_dim = z_dim
        self.channels = channels
        self.conv1 = nn.Conv2d(1, self.channels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.conv2 = nn.Conv2d(self.channels, self.channels, 4)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn2 = nn.BatchNorm2d(self.channels)
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.fc1 = nn.Linear(self.channels * 5 * 5, hidden_dim)
        self.fc2_loc_day = nn.Linear(hidden_dim, z_dim)
        self.fc2_scale_day = nn.Linear(hidden_dim, z_dim)

        self.fc2_loc_night = nn.Linear(hidden_dim, z_dim)
        self.fc2_scale_night = nn.Linear(hidden_dim, z_dim)

        self.softplus = nn.Softplus()

    def forward(self, viirs_observed, diurnality):
        assert viirs_observed.dim() == 3, f"viirs_observed.dim = {viirs_observed.dim()} != 3"

        viirs_observed = viirs_observed.unsqueeze(1)
        viirs_observed = self.pool(F.relu(self.bn1(self.conv1(viirs_observed))))
        viirs_observed = self.dropout1(viirs_observed)
        viirs_observed = self.pool(F.relu(self.bn2(self.conv2(viirs_observed))))
        viirs_observed = self.dropout2(viirs_observed)

        viirs_observed = viirs_observed.view(-1, self.channels * 5 * 5)

        hidden = self.softplus(self.fc1(viirs_observed))
        condition = (diurnality == 1).unsqueeze(-1)
        z_loc = torch.where(condition, self.fc2_loc_day(hidden), self.fc2_loc_night(hidden))
        z_scale = torch.exp(torch.where(condition, self.fc2_scale_day(hidden), self.fc2_scale_night(hidden)))
        return z_loc, z_scale


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.fc1 = nn.Linear(z_dim, rnn_dim)
        self.fc2_loc_day = nn.Linear(rnn_dim, z_dim)
        self.fc2_loc_night = nn.Linear(rnn_dim, z_dim)
        self.fc2_scale_day = nn.Linear(rnn_dim, z_dim)
        self.fc2_scale_night = nn.Linear(rnn_dim, z_dim)

    def forward(self, z_t_1, h_rnn, diurnality):
        h_combined = 0.5 * (torch.tanh(self.fc1(z_t_1)) + h_rnn)
        condition = (diurnality == 1).unsqueeze(-1)
        z_loc = torch.where(condition, self.fc2_loc_day(h_combined), self.fc2_loc_night(h_combined))
        z_scale = torch.where(condition, self.fc2_scale_day(h_combined), self.fc2_scale_night(h_combined))
        z_scale = F.softplus(z_scale)
        return z_loc, z_scale

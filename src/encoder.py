import torch
import torch.nn.functional as F
from torch import nn as nn


class GatedTransition(nn.Module):

    def __init__(self, z_dim, transition_dim):
        super().__init__()

        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)

        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

    def forward(self, z_t_1):
        _gate = torch.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))

        _proposed_mean = torch.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = F.softplus(self.lin_sig(torch.relu(proposed_mean)))

        return loc, scale


class Combiner(nn.Module):

    def __init__(self, z_dim, rnn_dim):
        super().__init__()

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


class ConvRNN(nn.Module):

    def __init__(self, image_dim, rnn_dim, rnn_layers, dropout_rate, use_lstm=True, channels=4):
        super().__init__()
        self.hidden_size = rnn_dim
        self.image_dim = image_dim
        self.channels = channels
        self.use_lstm = use_lstm
        input_dim = self.channels * 5 * 5
        rnn_dropout_rate = dropout_rate if rnn_layers > 1 else 0
        if self.use_lstm:
            print("Use LSTM")
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=rnn_dim, batch_first=True, bidirectional=False,
                               num_layers=rnn_layers, dropout=rnn_dropout_rate)
        else:
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                              batch_first=True, bidirectional=False, num_layers=rnn_layers,
                              dropout=rnn_dropout_rate)

        self.channels = channels
        self.conv1 = nn.Conv2d(1, self.channels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.conv2 = nn.Conv2d(self.channels, self.channels, 4)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn2 = nn.BatchNorm2d(self.channels)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, viirs_observed, hidden_state, cell_state=None):
        # viirs_observed: (batch, time, w, h)
        assert viirs_observed.dim() == 4, f"viirs_observed.dim() = {viirs_observed.dim()} != 4"
        assert viirs_observed.shape[2:] == self.image_dim
        assert viirs_observed.shape[1] in [5, 7]
        # process backwards in time
        viirs_observed = torch.flip(viirs_observed, dims=[1])

        z = viirs_observed.view(viirs_observed.shape[0] * viirs_observed.shape[1], 1,
                                self.image_dim[0], self.image_dim[1])

        z = self.pool(F.relu(self.bn1(self.conv1(z))))
        z = self.dropout1(z)
        z = self.pool(F.relu(self.bn2(self.conv2(z))))
        z = self.dropout2(z)
        z = z.view(viirs_observed.shape[0], viirs_observed.shape[1], self.channels * 5 * 5)

        if self.use_lstm:
            z, _ = self.rnn(z, (hidden_state, cell_state))
        else:
            z, _ = self.rnn(z, hidden_state)
        return torch.flip(z, dims=[1])

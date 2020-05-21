import torch
import torch.nn.functional as F
from torch import nn


class Decoder(nn.Module):
    def __init__(self, z_dim, channels=4, image_dimension=(30, 30), dropout_p=0.2):
        super().__init__()

        self.image_flatten_dim = image_dimension[0] * image_dimension[1]

        # decoder
        self.channels = channels
        self.d1 = nn.Linear(z_dim, self.channels * 8 * 8)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(self.channels, self.channels, 4)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(self.channels, 1, 3)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, z):
        z = F.relu(self.d1(z))
        z = z.view(-1, self.channels, 8, 8)
        z = self.leakyrelu(self.bn1(self.conv1(self.pd1(self.upsample1(z)))))
        z = self.dropout1(z)
        z = self.upsample2(z)
        z = self.pd2(z)
        z = self.conv2(z)
        z = z.reshape(-1, self.image_flatten_dim)

        return torch.sigmoid(z)

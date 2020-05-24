# This code is based on https://github.com/eriklindernoren/PyTorch-GAN of eriklindernoren

import torch.nn as nn
import torch
from torchsummary import summary


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 2, 2, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 3, 2, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 256, normalize=False)
        self.down2 = UNetDown(256, 512, dropout=0.25)
        self.down3 = UNetDown(512, 512, dropout=0.25)
        self.down4 = UNetDown(512, 512, normalize=False, dropout=0.25)

        self.up1 = UNetUp(512, 512, dropout=0.25)
        self.up2 = UNetUp(1024, 512, dropout=0.25)
        self.up3 = UNetUp(1024, 256)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, out_channels, 3, padding=1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)

        return self.final(u3)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=33):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 2, stride=2, padding=0)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 128, normalization=False),
            *discriminator_block(128, 256),
            nn.Conv2d(256, 1, 3, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator = GeneratorUNet().to(device)
# summary(generator, input_size=(32, 30, 30))
# discriminator = Discriminator().to(device)
# summary(discriminator, input_size=[(1, 30, 30), (32, 30, 30)])

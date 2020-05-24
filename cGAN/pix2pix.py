import argparse
import time
import datetime
import os
import sys
import matplotlib.pyplot as plt
import torch
from models import weights_init_normal, GeneratorUNet, Discriminator
from data import get_data_loader
from utils import set_fig_settings, FIG_REG_ASPECT_RATIO, FIG_REG_WIDTH
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="wildfires", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.75, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=30, help="size of image height")
parser.add_argument("--img_width", type=int, default=30, help="size of image width")
parser.add_argument('--target_hour', type=int, default=12, help="target hour to predict")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs("saved_models/%s" % args.dataset_name, exist_ok=True)
os.makedirs("sample_images/%s" % args.dataset_name, exist_ok=True)
os.makedirs("logs/%s" % args.dataset_name, exist_ok=True)
writer = SummaryWriter("logs/%s" % args.dataset_name)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss(reduction='none')
# criterion_GAN = torch.nn.BCEWithLogitsLoss(reduction='none')

# criterion_pixelwise = torch.nn.MSELoss(reduction='none')
criterion_pixelwise = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.full((30, 30), 10))
# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 1
lambda_GAN = 1

# Calculate output of image discriminator (PatchGAN)
patch = (1, args.img_height // 2 ** 2, args.img_width // 2 ** 2)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    criterion_GAN.to(device)
    criterion_pixelwise.to(device)

if args.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (args.dataset_name, args.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (args.dataset_name, args.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

dataloader, val_dataloader = get_data_loader(args.batch_size, args.target_hour)


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = imgs[0].to(device)
    real_B = imgs[1].to(device)
    fake_B = generator(real_A)

    img_dir = "sample_images/%s/%s.png" % (args.dataset_name, batches_done)
    set_fig_settings((FIG_REG_WIDTH * 2, FIG_REG_WIDTH * 1.25))
    fig = plt.figure()
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title('Real') if i == 0 else plt.title('Fake')
        plt.imshow([real_B[0, 0].cpu().detach().numpy(), fake_B[0, 0].cpu().detach().numpy()][i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_dir)


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(args.epoch, args.n_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = batch[0].to(device)
        real_B = batch[1].to(device)
        sample_weights = batch[2].view(real_A.size(0), 1, 1, 1).to(device)
        # sample_weights = torch.full_like(sample_weights, 1, device=device)

        # Adversarial ground truths
        valid = torch.full((real_A.size(0), *patch), 1, device=device)
        fake = torch.full((real_A.size(0), *patch), 0, device=device)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid) * sample_weights
        loss_GAN = loss_GAN.mean()

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B) * sample_weights
        loss_pixel = loss_pixel.mean()

        # Total loss
        loss_G = lambda_GAN * loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid) * sample_weights
        loss_real = loss_real.mean()

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake) * sample_weights
        loss_fake = loss_fake.mean()

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = args.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (epoch, args.n_epochs, i, len(dataloader),
               loss_D.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item(),
               time_left)
        )

        writer.add_scalars('losses', {'g_loss': loss_G.item()}, batches_done)
        writer.add_scalars('losses', {'d_loss': loss_D.item()}, batches_done)
        writer.add_scalars('losses', {'gan_loss': loss_GAN.item()}, batches_done)
        writer.add_scalars('losses2', {'pixel_loss': loss_pixel.item()}, batches_done)

        # If at sample interval save image
        if batches_done % args.sample_interval == 0:
            sample_images(batches_done)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (args.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (args.dataset_name, epoch))

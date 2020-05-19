import argparse
from pathlib import Path

import numpy as np
import pyro
import torch
import visdom
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader

from dataset import VIIRSDataset
from src.vae_plots import plot_trajectory
from src.vae_plots import viirs_tsne, plot_llk_viirs
from src.vae_viirs import VAE


def main(args):
    # clear param store
    csv_path = Path("vae_results/VAE_viirs_embedding.csv")
    output_path = Path("vae_results/transisition.png")
    pyro.clear_param_store()

    # setup MNIST data loaders
    # train_loader, test_loader
    viirs_dataset = VIIRSDataset(load_records=args.load_records)
    train_loader = DataLoader(viirs_dataset, batch_size=256, shuffle=True, num_workers=4)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    # setup visdom for visualization
    if args.visdom_flag:
        vis = visdom.Visdom()

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch == args.tsne_iter:
            viirs_tsne(vae=vae, data_loader=train_loader)
            plot_llk_viirs(np.array(train_elbo))

    plot_trajectory(csv_path, output_path)

    return vae


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.1')
    np.random.seed(0)
    pyro.set_rng_seed(0)
    torch.manual_seed(0)

    # parse command line arguments
    epoch = 150
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--load-records', default=100, type=int, help='number of record to load')
    parser.add_argument('-n', '--num-epochs', default=epoch + 1, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-4, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=True, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
    parser.add_argument('-i-tsne', '--tsne_iter', default=epoch, type=int, help='epoch when tsne visualization runs')
    args = parser.parse_args()

    model = main(args)

# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from src.visualize.vae_plots import plot_trajectory

if __name__ == '__main__':
    csv_path = Path("vae_results/VAE_viirs_embedding.csv")
    output_path = Path("vae_results/transisition.png")
    plot_trajectory(csv_path, output_path)

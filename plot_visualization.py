# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from dataset import WildFireDataset

if __name__ == '__main__':
    csv_path = Path("vae_results/VAE_viirs_embedding.csv")
    output_path = Path("vae_results/transisition.png")

    df = pd.read_csv(csv_path, index_col=0, dtype={"index": int})

    print(df.keys(), df.dtypes)
    figure = plt.figure(figsize=(14, 12), num=0)
    time_step = [-48, -36, -24, -12, 0, 12, 24]
    df = df.sample(frac=1)

    cmap = cm.get_cmap("Greens")

    x_lim = (df[['-48_h_0', '-36_h_0', '-24_h_0', '-12_h_0', '0_h_0', '12_h_0', '24_h_0']].min().min(),
             df[['-48_h_0', '-36_h_0', '-24_h_0', '-12_h_0', '0_h_0', '12_h_0', '24_h_0']].max().max())

    y_lim = (df[['-48_h_1', '-36_h_1', '-24_h_1', '-12_h_1', '0_h_1', '12_h_1', '24_h_1']].min().min(),
             df[['-48_h_1', '-36_h_1', '-24_h_1', '-12_h_1', '0_h_1', '12_h_1', '24_h_1']].max().max())
    print(x_lim, y_lim)
    df = df[:36]
    df = df.sort_values(by="index")

    wildfire = WildFireDataset(load_records=df["index"].max() + 1)
    idx = 0

    for _, row in df.iterrows():
        plt.figure(num=0)

        plt.subplot(6, 6, idx + 1)
        idx += 1
        for j, (t1, t2) in enumerate(zip(time_step[:-1], time_step[1:])):
            plt.arrow(row[f'{t1}_h_0'], row[f'{t1}_h_1'],
                      row[f'{t2}_h_0'] - row[f'{t1}_h_0'],
                      row[f'{t2}_h_1'] - row[f'{t1}_h_1']
                      , color=cmap(j / 6)[:3]
                      , width=0.3)
            # , linewidth=((j + 1) * 0.4) ** 2)
        plt.hlines(-20, 0, x_lim[1])
        plt.vlines(0, -20, y_lim[1])
        plt.text(x_lim[0], y_lim[0], int(row["index"]))
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.gca().get_xaxis().set_ticklabels([])
        plt.gca().get_yaxis().set_ticklabels([])

        plt.scatter(row[f'{time_step[0]}_h_0'], row[f'{time_step[0]}_h_1'], marker="^", s=40, color="C0")
        plt.scatter(row[f'{time_step[4]}_h_0'], row[f'{time_step[4]}_h_1'], marker="o", s=40, color="C0")
        plt.scatter(row[f'{time_step[-1]}_h_0'], row[f'{time_step[-1]}_h_1'], marker="s", s=40, color="C1")
        plt.grid()

        plt.figure(1)
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.title(f'{-12 * (4 - i)} hours')
            plt.imshow(wildfire[int(row["index"])][0][4 - i])
            plt.axis('off')

        # Plt Y detections
        for i in range(2):
            plt.subplot(2, 5, i + 5 + 1)
            plt.title(f'+{12 * (i + 1)} hours')
            plt.imshow(wildfire[int(row["index"])][1][i])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(Path(f"vae_results/details_{int(row['index'])}.png"))
        plt.close(1)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(output_path)

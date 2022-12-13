#!/usr/bin/env python3

import os.path as path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use(path.join("styles", "eccv.mplstyle"))
fig = plt.figure(figsize=(1.4, 1.4))
ax = fig.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
fig.set_tight_layout(False)
fig.subplots_adjust(left=0.1, right=0.9, top=0.87, bottom=0.07, hspace=0.4, wspace=0.4)

for ax_i in ax:
    for ax_ij in ax_i:
        ax_ij.axhline(0.0, color="lightgrey", linewidth=1.0)
        ax_ij.axvline(0.0, color="lightgrey", linewidth=1.0)
        ax_ij.set_aspect("equal")
        ax_ij.set(xlim=(-2.1, 2.1), ylim=(-2.1, 2.1), xticks=[], yticks=[])

title_kwargs = dict(fontsize="medium", pad=3)

ax[0][0].plot(
    [-10.0, -0.5, -0.5, 0.5, 0.5, 10.0],
    [-10.0, -0.5, 0.0, 0.0, 0.5, 10.0],
    color="C1",
)
ax[0][0].set_title("Ours", **title_kwargs)

steps = np.arange(start=-50.0, stop=50.0)
ax[0][1].step(steps, steps, where="mid", color="C2")
ax[0][1].set_title("Rounding", **title_kwargs)

steps = np.arange(start=0.5, stop=10, step=0.3)
ax[1][0].step(
    np.concatenate([-steps[::-1], [-0.5, 0.5], steps]),
    np.concatenate([-steps[::-1], [0.0, 0.0], steps]),
    where="mid",
    color="C3",
)
ax[1][0].set_title("Quantized", **title_kwargs)

steps = 2.0 ** np.arange(start=-1.0, stop=5.0, step=0.4)
ax[1][1].step(
    np.concatenate([-steps[::-1], [-0.5, 0.5], steps]),
    np.concatenate([-steps[::-1], [0.0, 0.0], steps]),
    where="mid",
    color="C4",
)
ax[1][1].set_title("Nonuniform", **title_kwargs)

fig.savefig(path.join("figures", "policy_functions.pdf"))

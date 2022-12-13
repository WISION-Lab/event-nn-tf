#!/usr/bin/env python3

import os.path as path

import matplotlib.pyplot as plt
import numpy as np

# Commented lines are for the paper version of the figure.

# plt.style.use(path.join("styles", "eccv.mplstyle"))
# fig = plt.figure(figsize=(4.5, 1.3))

# Quadruple the figure size when inserting into the poster.
plt.style.use(path.join("styles", "poster.mplstyle"))
fig = plt.figure(figsize=(3.2, 1.2))
axs = fig.subplots(ncols=2)

results_openpose_conventional = np.array([[0.0, 0.7640, 7.055e10]])
results_openpose_event = np.array(
    [
        [0.01, 0.7656, 1.071e10],
        [0.02, 0.7696, 8.825e9],
        [0.04, 0.7637, 6.780e9],
        [0.06, 0.7448, 5.640e9],
        [0.08, 0.7310, 4.890e9],
    ]
)
results_openpose_skip_conv_0 = np.array(
    [
        [0.01, 0.7603, 1.027e10],
        [0.02, 0.7277, 7.873e9],
        [0.04, 0.6644, 5.837e9],
    ]
)
results_openpose_skip_conv_8 = np.array(
    [
        [0.01, 0.7621, 1.092e10],
        [0.02, 0.7311, 8.816e9],
        [0.04, 0.6635, 7.054e9],
    ]
)

for i, (results, name, color) in enumerate(
    (
        (results_openpose_conventional, "Conventional", "C0"),
        (results_openpose_skip_conv_0, "Skip-Conv", "C2"),
        (results_openpose_skip_conv_8, "Skip-Conv Reset", "C3"),
        (results_openpose_event, "Ours", "C1"),
    )
):
    axs[0].plot(results[:, 2], results[:, 1], color=color, zorder=-i)
    axs[0].scatter(results[:, 2], results[:, 1], s=10.0, color=color, label=name, zorder=-i)
axs[0].set(
    # ylabel="PCK",
    ylabel="Accuracy (PCK)",
    xscale="log",
    # xlim=(None, 1e11),
    xlim=(3.5e9, 1e11),
    ylim=(0.64, 0.78),
    # yticks=np.arange(0.64, 0.79, 0.02),
    yticks=np.arange(0.62, 0.79, 0.04),
)
# axs[0].set_xlabel("Ops", labelpad=-9, loc="left")
axs[0].set_xlabel("Ops", labelpad=-7, loc="left")
# axs[0].set_title("(a) Video Pose Estimation", fontsize="medium")
axs[0].set_title("Video Pose Estimation", fontsize="medium")

results_yolo_conventional = np.array([[0.0, 0.5538, 1.537e10]])
results_yolo_event = np.array(
    [
        [0.01, 0.5545, 7.472e9],
        [0.02, 0.5563, 6.074e9],
        [0.04, 0.5618, 4.517e9],
        [0.08, 0.5619, 3.061e9],
        [0.12, 0.5463, 2.306e9],
        [0.16, 0.5024, 1.812e9],
    ]
)
results_yolo_skip_conv_0 = np.array(
    [
        [0.01, 0.5413, 7.340e9],
        [0.02, 0.4581, 5.705e9],
        [0.04, 0.3098, 3.819e9],
    ]
)
results_yolo_skip_conv_8 = np.array(
    [
        [0.01, 0.5406, 8.111e9],
        [0.02, 0.4544, 6.692e9],
        [0.04, 0.2737, 5.054e9],
    ]
)

for i, (results, name, color) in enumerate(
    [
        (results_yolo_conventional, "Conventional", "C0"),
        (results_yolo_skip_conv_0, "Skip-Conv", "C2"),
        (results_yolo_skip_conv_8, "Skip-Conv-8", "C3"),
        (results_yolo_event, "Ours", "C1"),
    ]
):
    axs[1].plot(results[:, 2], results[:, 1], color=color, zorder=-i)
    axs[1].scatter(results[:, 2], results[:, 1], s=10.0, color=color, label=name, zorder=-i)
axs[1].set(
    # ylabel="mAP50",
    ylabel="Accuracy (mAP50)",
    xscale="log",
    xlim=(1.5e9, 2e10),
    ylim=(0.25, 0.6),
    # yticks=np.arange(0.25, 0.61, 0.05),
    yticks=np.arange(0.2, 0.61, 0.1),
)
axs[1].set_xticklabels([], minor=True)
# axs[1].set_xlabel("Ops", labelpad=-9, loc="left")
axs[1].set_xlabel("Ops", labelpad=-7, loc="left")
# axs[1].set_title("(b) Video Object Detection", fontsize="medium")
axs[1].set_title("Video Object Detection", fontsize="medium")
# axs[1].legend(loc="center left", bbox_to_anchor=(1.08, 0.5))
axs[1].legend(loc="lower left", framealpha=1.0, bbox_to_anchor=(0.7, 0.05))

# fig.savefig(path.join("figures", "pareto_curves.pdf"))
fig.savefig(path.join("figures", "pareto_curves_poster.pdf"))

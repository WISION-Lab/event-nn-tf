#!/usr/bin/env python3

import os.path as path
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

with open(path.join("outputs", "jhmdb_openpose_time_conventional.p"), "rb") as f:
    results_conventional = pickle.load(f)
with open(path.join("outputs", "jhmdb_openpose_time_event.p"), "rb") as f:
    results_event = pickle.load(f)
with open(path.join("outputs", "jhmdb_openpose_time_memory_loss.p"), "rb") as f:
    results_memory_loss = pickle.load(f)

# Commented lines are for the paper version of the figure.

# plt.style.use(path.join("styles", "eccv.mplstyle"))
# fig = plt.figure(figsize=(2.4, 1.3))

# Quadruple the figure size when inserting into the poster.
plt.style.use(path.join("styles", "poster.mplstyle"))
fig = plt.figure(figsize=(1.8, 1.4))
ax = fig.subplots()

ax.plot(results_conventional["pck_single_person"], label="Conventional")
ax.plot(results_event["pck_single_person"], label="Event")
# ax.plot(results_memory_loss["pck_single_person"], label="Without Long-Term Memory")
ax.plot(results_memory_loss["pck_single_person"], label="Without LTM")
ax.legend()
# ax.set(ylim=(0.0, 1.0), xticks=[0, 40], yticks=[0.0, 1.0])
ax.set(ylim=(0.0, 1.0))
# ax.set_xlabel("Time Step", labelpad=-6)
ax.set_xlabel("Time Step")
# ax.set_ylabel("PCK", labelpad=-4)
ax.set_ylabel("Accuracy (PCK)")
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(0.2))

# fig.savefig(path.join("figures", "accuracy_over_time.pdf"))
fig.savefig(path.join("figures", "accuracy_over_time_poster.pdf"))

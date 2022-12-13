import os
import os.path as path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from utils.misc import find_unused_dirname, rescale_video, visual_as_float


def delta_cmap(delta, cmap="RdBu", scale=1.0):
    max_abs = np.max(np.abs(delta))
    if max_abs == 0:
        delta = np.full_like(delta, 0.5)
    else:
        delta = 0.5 * (1 + delta / max_abs)
    cmap_output = plt.get_cmap(cmap)(delta)[..., :3]
    return rescale_video(cmap_output, scale=scale, method="nearest")


def delta_mask(
    video, deltas, masks, cmap_delta="RdBu", cmap_mask="Greys", border_size=2, spacer_size=15
):
    columns = [visual_as_float(video[..., :3])]
    for column_deltas, column_masks in zip(deltas, masks):
        column = []
        for i, (delta, mask) in enumerate(zip(column_deltas, column_masks)):
            delta = delta_cmap(delta, cmap=cmap_delta)
            mask = mask_cmap(mask, cmap=cmap_mask)
            buffer = _make_buffer(delta, border_size, axis=2)
            column.append(np.concatenate([delta, buffer, mask], axis=2))
            if i < len(column_deltas) - 1:
                column.append(_make_buffer(column[-1], border_size, axis=1))
        columns.append(np.concatenate(column, axis=1))

    max_height = max(column.shape[1] for column in columns)
    columns_padded = []
    for i, column in enumerate(columns):
        pad_top = (max_height - column.shape[1]) // 2
        pad_bottom = max_height - column.shape[1] - pad_top
        buffer_top = _make_buffer(column, pad_top, axis=1)
        buffer_bottom = _make_buffer(column, pad_bottom, axis=1)
        columns_padded.append(np.concatenate([buffer_top, column, buffer_bottom], axis=1))
        if i < len(columns) - 1:
            columns_padded.append(_make_buffer(columns_padded[-1], spacer_size, axis=2))
    return np.concatenate(columns_padded, axis=2)


def mask_cmap(mask, cmap="Greys", scale=1.0):
    cmap_output = plt.get_cmap(cmap)(mask)[..., :3]
    return rescale_video(cmap_output, scale=scale, method="nearest")


def op_bar(video, ops_array, max_ops, bar_width=50, color=(0, 255, 0)):
    output = []
    for t, frame in enumerate(video):
        frame = video[t]
        height, width, channels = frame.shape
        frame_out = np.zeros((height, width + bar_width, channels), frame.dtype)
        frame_out[:, :width] = frame
        bar_height = int(ops_array[t] / max_ops * height)
        if bar_height > 0:
            frame_out[-bar_height:, -bar_width:] = color
        output.append(frame_out)
    return np.array(output)


def op_line(
    series,
    filename,
    colors=None,
    labels=None,
    ylim=(0.0, 1.0),
    labelpad=-20,
    figsize=(12.0, 3.0),
    legend_loc="upper right",
):
    image_dir = find_unused_dirname(path.splitext(filename)[0])
    os.makedirs(image_dir)
    try:
        for t in range(len(series[0])):
            fig = plt.figure(figsize=figsize)
            ax = fig.subplots()
            for i, series_i in enumerate(series):
                label = None if (labels is None) else labels[i]
                color = None if (colors is None) else colors[i]
                ax.plot(series_i[: t + 1], color=color, label=label)
                ax.scatter(t, series_i[t], color=color)
            if labels is not None:
                ax.legend(loc=legend_loc)
            ax.set(xlim=(0, len(series[0])), ylim=ylim, xticks=[], yticks=list(ylim))
            ax.set_ylabel("Cost Fraction", labelpad=labelpad)
            fig.savefig(path.join(image_dir, "{}.jpg".format(t)))
            plt.close(fig)
        subprocess.call(["ffmpeg", "-y", "-i", path.join(image_dir, "%d.jpg"), filename])
    finally:
        shutil.rmtree(image_dir)


def _make_buffer(base_array, size, axis):
    shape = list(base_array.shape)
    shape[axis] = size
    return np.zeros(shape, base_array.dtype)

import os.path as path
import json

from utils.misc import (
    compute_padded_coordinates,
    listdir_filtered,
    prepare_padded_video_data,
    prepare_ragged_data,
)

IMAGENET_VOD_NAMES = [
    "airplane",
    "antelope",
    "bear",
    "bicycle",
    "bird",
    "bus",
    "car",
    "cattle",
    "dog",
    "domestic_cat",
    "elephant",
    "fox",
    "giant_panda",
    "hamster",
    "horse",
    "lion",
    "lizard",
    "monkey",
    "motorcycle",
    "rabbit",
    "red_panda",
    "sheep",
    "snake",
    "squirrel",
    "tiger",
    "train",
    "turtle",
    "watercraft",
    "whale",
    "zebra",
]

def load_data(data_dir, split, size=(224, 384), preprocess_func=None):
    with open(path.join(data_dir, "{}.json".format(split)), "r") as f:
        json_data = json.load(f)

    frames = []
    boxes = []
    classes = []
    n_annotations = []
    for video_id, item in json_data.items():
        frames_i = listdir_filtered(path.join(data_dir, "frames", video_id), ".jpg")
        frames.append(frames_i)

        boxes_i = []
        classes_i = []
        n_annotations_i = []
        for frame in item:
            size_before = frame["height"], frame["width"]
            boxes_t = []
            classes_t = []
            for annotation in frame["annotations"]:
                y_1, x_1, y_2, x_2 = annotation["box"]
                x_1, y_1 = compute_padded_coordinates(x_1, y_1, size_before, size)
                x_2, y_2 = compute_padded_coordinates(x_2, y_2, size_before, size)
                boxes_t.append([y_1, x_1, y_2, x_2])
                classes_t.append(annotation["category"])
            boxes_i.append(boxes_t)
            classes_i.append(classes_t)
            n_annotations_i.append(len(frame["annotations"]))

        empty_box = [float("nan")] * 4
        boxes_i_padded = []
        classes_i_padded = []
        for boxes_t, classes_t in zip(boxes_i, classes_i):
            n_pad = max(n_annotations_i) - len(boxes_t)
            boxes_i_padded.append(boxes_t + [empty_box] * n_pad)
            classes_i_padded.append(classes_t + [-1] * n_pad)
        boxes.append(boxes_i_padded)
        classes.append(classes_i_padded)
        n_annotations.append(n_annotations_i)

    n_items = len(frames)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    frames = prepare_padded_video_data(frames, size, preprocess_func=preprocess_func)

    # Dataset shape (1, n_frames, n_boxes, 4)
    boxes = prepare_ragged_data(boxes, inner_shape=(4,))

    # Dataset shape (1, n_frames, n_boxes)
    classes = prepare_ragged_data(classes)

    # Dataset shape (1, n_frames)
    n_annotations = prepare_ragged_data(n_annotations)

    return (frames, boxes, classes, n_annotations), n_items

import json
from os import path as path

from models.yolo import preprocess_image
from utils.misc import (
    compute_padded_coordinates,
    listdir_filtered,
    prepare_padded_video_data,
    prepare_ragged_data,
)

CATEGORY_NAMES = [
    "person",
    "giant_panda",
    "lizard",
    "parrot",
    "skateboard",
    "sedan",
    "ape",
    "dog",
    "snake",
    "monkey",
    "hand",
    "rabbit",
    "duck",
    "cat",
    "cow",
    "fish",
    "train",
    "horse",
    "turtle",
    "bear",
    "motorbike",
    "giraffe",
    "leopard",
    "fox",
    "deer",
    "owl",
    "surfboard",
    "airplane",
    "truck",
    "zebra",
    "tiger",
    "elephant",
    "snowboard",
    "boat",
    "shark",
    "mouse",
    "frog",
    "eagle",
    "earless_seal",
    "tennis_racket",
]

MAJOR_CAMERA_MOTION_TEST = [
    "0070461469",
    "00fef116ee",
    "02caec8ac0",
    "05e73c3ecb",
    "0c6d13ee2c",
    "0c7ba00455",
    "0cba3e52eb",
    "13006c4c7e",
    "18f1e2f716",
    "1a1dc21969",
    "1e0c2e54f2",
    "1e458b1539",
    "1e790eae99",
    "1f2015e056",
    "25fc493839",
    "29b87846e7",
    "37e0b4642b",
    "3bde1da2cf",
    "3c5f4e6672",
    "3c80682cc6",
    "3d6a761295",
    "3da878c317",
    "4015a1e1cc",
    "406cd7bd48",
    "407b87ba26",
    "40a5628dcc",
    "4566034eaf",
    "4595935b88",
    "4be9025726",
    "4c269afea9",
    "4e1ef26a1e",
    "50f89963c0",
    "544b1486ac",
    "5d0b35f30f",
    "5e130392e1",
    "5e75de78ae",
    "673b1c03c9",
    "67780f49c2",
    "6d6b09b420",
    "6e89b7359d",
    "6e9feafa2b",
    "6f9019f0ea",
    "759d8249dd",
    "78e639c2c4",
    "7c8014406a",
    "7d04e540f5",
    "7d5df020bf",
    "7e6a27cc7c",
    "851f73e4fc",
    "87e1975888",
    "87f5d4903c",
    "8a99d63296",
    "8b0697f61a",
    "8cb68f36f6",
    "8fa5b3778f",
    "90c2c5c336",
    "9124189275",
    "9584210f86",
    "9b5aa49509",
    "a51335af59",
    "a5b71f76fb",
    "af658a277c",
    "b0313efe37",
    "b3cee57f31",
    "b5d3b9be03",
    "b8a2104720",
    "b8d6f92a65",
    "bbdc38baa0",
    "c64e9d1f7e",
    "cba3e31e88",
    "cddf78284d",
    "ce45145721",
    "cff0bbcba8",
    "d1ed509b94",
    "d7b3892660",
    "dc0928b157",
    "deb31e46cf",
    "e1fc6d5237",
    "e228ce16fd",
    "e60826ddf9",
    "e98d115b9c",
    "eb92c92912",
    "eec8403cc8",
    "ef22b8a227",
    "f4865471b4",
    "f94cd39525",
    "fbdda5ec7b",
    "fc0db37221",
]

MINOR_CAMERA_MOTION_TEST = [
    "047436c72c",
    "0e1f91c0d7",
    "0ef454b3f0",
    "11444b16da",
    "11b3298d6a",
    "1655f4782a",
    "16608ccef6",
    "2462c51412",
    "2cf6c4d17e",
    "386b050bd0",
    "3db571b7ee",
    "4be0ac97df",
    "4c18a7bfab",
    "51ee638399",
    "835aea9584",
    "88b84fe107",
    "9246556dfd",
    "cfd1e8166f",
    "d380084b7c",
    "ed33e8efb7",
    "fa666fcc95",
    "fb25b14e48",
]

NO_CAMERA_MOTION_TEST = [
    "26d8d48248",
    "51d60e4f93",
    "9fca469ddd",
    "d380084b7c",
]


def load_data(
    data_dir, split, size=(288, 512), n_frames_filter=None, video_id_filter=None, max_frames=100
):
    if split.startswith("auto"):
        return _load_auto_data(data_dir, split, size, n_frames_filter, video_id_filter, max_frames)
    elif split == "train":
        return _load_train_data(data_dir, size, n_frames_filter, video_id_filter, max_frames)
    elif split in ["test", "val"]:
        return _load_unlabeled_data(
            data_dir, split, size, n_frames_filter, video_id_filter, max_frames
        )
    else:
        raise ValueError("Invalid split name {}.".format(split))


def _load_auto_data(data_dir, split, size, n_frames_filter, video_id_filter, max_frames):
    base_dir = path.join(data_dir, split)
    with open(path.join(base_dir, "data.json"), "r") as f:
        json_data = json.load(f)

    frames = []
    boxes = []
    classes = []
    n_annotations = []
    for item in json_data:
        frames_i = listdir_filtered(path.join(base_dir, "frames", item["video_id"]), ".jpg")
        if (video_id_filter is not None) and (item["video_id"] not in video_id_filter):
            continue
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        if max_frames and len(frames_i) > max_frames:
            continue
        frames.append(frames_i)

        _process_annotations(item, boxes, classes, n_annotations, size)

    n_items = len(frames)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    frames = prepare_padded_video_data(frames, size, preprocess_func=preprocess_image)

    # Dataset shape (1, n_frames, n_boxes, 4)
    boxes = prepare_ragged_data(boxes, inner_shape=(4,))

    # Dataset shape (1, n_frames, n_boxes)
    classes = prepare_ragged_data(classes)

    # Dataset shape (1, n_frames)
    n_annotations = prepare_ragged_data(n_annotations)

    return (frames, boxes, classes, n_annotations), n_items


def _load_train_data(data_dir, size, n_frames_filter, video_id_filter, max_frames):
    base_dir = path.join(data_dir, "train")
    with open(path.join(base_dir, "data.json"), "r") as f:
        json_data = json.load(f)

    frames = []
    frame_masks = []
    boxes = []
    classes = []
    n_annotations = []
    for item in json_data:
        frames_i = listdir_filtered(path.join(base_dir, "frames", item["video_id"]), ".jpg")
        if (video_id_filter is not None) and (item["video_id"] not in video_id_filter):
            continue
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        if max_frames and len(frames_i) > max_frames:
            continue
        frames.append(frames_i)

        base_names_i = [path.basename(x) for x in frames_i]
        labelled_i = [(x in item["labelled_frames"]) for x in base_names_i]
        frame_masks.append(labelled_i)

        _process_annotations(item, boxes, classes, n_annotations, size)

    n_items = len(frames)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    frames = prepare_padded_video_data(frames, size, preprocess_func=preprocess_image)

    # Dataset shape (1, n_frames)
    frame_masks = prepare_ragged_data(frame_masks)

    # Dataset shape (1, n_frames, n_boxes, 4)
    boxes = prepare_ragged_data(boxes, inner_shape=(4,))

    # Dataset shape (1, n_frames, n_boxes)
    classes = prepare_ragged_data(classes)

    # Dataset shape (1, n_frames)
    n_annotations = prepare_ragged_data(n_annotations)

    return (frames, frame_masks, boxes, classes, n_annotations), n_items


def _load_unlabeled_data(data_dir, split, size, n_frames_filter, video_id_filter, max_frames):
    base_dir = path.join(data_dir, split)
    with open(path.join(base_dir, "data.json"), "r") as f:
        json_data = json.load(f)

    frames = []
    for item in json_data:
        frames_i = listdir_filtered(path.join(base_dir, "frames", item["video_id"]), ".jpg")
        if (video_id_filter is not None) and (item["video_id"] not in video_id_filter):
            continue
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        if max_frames and len(frames_i) > max_frames:
            continue
        frames.append(frames_i)

    n_items = len(frames)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    frames = prepare_padded_video_data(frames, size, preprocess_func=preprocess_image)

    return frames, n_items


def _process_annotations(item, boxes, classes, n_annotations, size):
    size_before = item["height"], item["width"]
    boxes_i = []
    classes_i = []
    max_annotations = 0
    for annotations_t in item["annotations"]:
        boxes_t = []
        classes_t = []
        for annotation in annotations_t:
            y_1, x_1, y_2, x_2 = annotation["box"]
            x_1, y_1 = compute_padded_coordinates(x_1, y_1, size_before, size)
            x_2, y_2 = compute_padded_coordinates(x_2, y_2, size_before, size)
            boxes_t.append([y_1, x_1, y_2, x_2])
            classes_t.append(annotation["category"])
        max_annotations = max(max_annotations, len(annotations_t))
        boxes_i.append(boxes_t)
        classes_i.append(classes_t)

    empty_box = [float("nan")] * 4
    boxes_i_padded = []
    classes_i_padded = []
    for boxes_t, classes_t in zip(boxes_i, classes_i):
        n_pad = max_annotations - len(boxes_t)
        boxes_i_padded.append(boxes_t + [empty_box] * n_pad)
        classes_i_padded.append(classes_t + [-1] * n_pad)
    boxes.append(boxes_i_padded)
    classes.append(classes_i_padded)

    n_annotations_i = []
    for annotations_t in item["annotations"]:
        n_annotations_i.append(len(annotations_t))
    n_annotations.append(n_annotations_i)

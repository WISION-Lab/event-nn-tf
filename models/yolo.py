import os
import os.path as path
import sys

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

from eventnn.layers import Accumulator, Bias, Fuse, Gate, LayerWrapper, Mask, Unmask
from eventnn.model import EventModel
from utils.misc import (
    rescale_image,
    resize_image_to_multiple,
    scale_coordinates,
    visual_as_float,
    visual_as_uint8,
)

# Resources consulted:
# https://arxiv.org/abs/1804.02767
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
# https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/models.py
# https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/utils.py

# noinspection SpellCheckingInspection
COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic_light",
    "fire_hydrant",
    "stop_sign",
    "parking_meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports_ball",
    "kite",
    "baseball_bat",
    "baseball_glove",
    "skateboard",
    "surfboard",
    "tennis_racket",
    "bottle",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot_dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell_phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy_bear",
    "hair_drier",
    "toothbrush",
]


# Dumps predictions to text files so they can be consumed by an
# open-source evaluation toolkit.
# Very much an abuse of the tf.Metric interface...
class FileDump:
    # noinspection PyDefaultArgument
    def __init__(
        self,
        base_dir,
        max_boxes=100,
        iou_threshold=0.4,
        score_threshold=0.7,
        class_names=COCO_NAMES,
        verbose=False,
        name=None,
    ):
        self.base_dir = base_dir
        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.class_names = class_names
        self.verbose = verbose
        self.name = "file_dump" if (name is None) else name
        self.file_index = 0

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {
            "base_dir": self.base_dir,
            "max_boxes": self.max_boxes,
            "iou_threshold": self.iou_threshold,
            "score_threshold": self.score_threshold,
            "class_names": self.class_names,
            "verbose": self.verbose,
        }

    def reset_states(self):
        self.file_index = 0

    # noinspection PyMethodMayBeStatic
    def result(self):
        return 0.0

    def update_state(self, y_true, y_pred):
        true_dir = path.join(self.base_dir, "true")
        pred_dir = path.join(self.base_dir, "pred")
        os.makedirs(true_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        for i in range(y_pred[0].shape[0]):
            boxes_pred, scores_pred, classes_pred = postprocess_image(
                y_pred[0][i],
                y_pred[1][i],
                max_boxes=self.max_boxes,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
            )
            with open(path.join(pred_dir, "{}.txt".format(self.file_index)), "w") as f:
                for box_pred, score_pred, class_pred in zip(boxes_pred, scores_pred, classes_pred):
                    x_1, y_1, w, h = _box_to_int_xywh(box_pred)
                    name = self.class_names[class_pred]
                    f.write("{} {} {} {} {} {}\n".format(name, score_pred, x_1, y_1, w, h))

            n_annotations_true = y_true[2][i]
            boxes_true = y_true[0][i][:n_annotations_true]
            classes_true = y_true[1][i][:n_annotations_true]
            with open(path.join(true_dir, "{}.txt".format(self.file_index)), "w") as f:
                for box_true, class_true in zip(boxes_true, classes_true):
                    x_1, y_1, w, h = _box_to_int_xywh(box_true)
                    name = self.class_names[class_true]
                    f.write("{} {} {} {} {}\n".format(name, x_1, y_1, w, h))

            if self.verbose:
                print(
                    "Finished dumping output {}.".format(self.file_index),
                    file=sys.stderr,
                    flush=True,
                )

            self.file_index += 1


def apply_bn_gamma(model, input_range=1.0):
    for layer in model.gates:
        adjustments = layer.policy.adjustments
        if layer.name == "input_gate":
            # 1 / sqrt(12) is the standard deviation of a uniform
            # distribution between 0 and 1.
            adjustments.assign(input_range * tf.ones_like(adjustments) / tf.sqrt(12.0))
        else:
            bn_layer = model.get_layer("bn_wrapper_{}".format(id(layer))).layer
            gamma = tf.abs(bn_layer.gamma)  # Gamma may be negative.
            reshape = [1] * len(adjustments.shape)
            for axis, size in zip(bn_layer.axis, gamma.shape):
                reshape[axis - 1] = size
            gamma = tf.reshape(gamma, reshape)
            for axis in range(len(reshape)):
                repeats = adjustments.shape[axis] // reshape[axis]
                gamma = tf.repeat(gamma, repeats, axis=axis)
            adjustments.assign(gamma)


def postprocess_image(boxes, classes, max_boxes=100, iou_threshold=0.4, score_threshold=0.7):
    selected = tf.image.non_max_suppression(
        boxes[..., :4],
        boxes[..., 4],
        max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    selected = np.array(selected)
    boxes = np.array(boxes)[selected]
    classes = np.array(classes)[selected]
    coordinates = boxes[:, :4]
    scores = boxes[:, 4]
    return coordinates, scores, classes


def postprocess_video(boxes, classes):
    return [postprocess_image(boxes_t, classes_t) for boxes_t, classes_t in zip(boxes, classes)]


def preprocess_image(image):
    image = visual_as_float(image)
    image = resize_image_to_multiple(image, 32)
    return image


def preprocess_video(video):
    return tf.stack([preprocess_image(frame) for frame in video])


# Based on my implementation at https://github.com/mattdutson/yolo
def yolo_v3(input_size, batch_size=1, n_classes=80, darknet_weights=None, h5_weights=None):
    inputs = Input(batch_input_shape=(batch_size,) + input_size + (3,))
    x = inputs

    if darknet_weights is not None:
        w_darknet = open(darknet_weights, "rb")
        np.fromfile(w_darknet, dtype=np.int32, count=5)  # Skip file header
    else:
        w_darknet = None

    if h5_weights is not None:
        w_h5 = h5py.File(h5_weights, "r")
    else:
        w_h5 = None

    # Model input
    x = Mask()(x)
    x = Gate(name="input_gate")(x)

    i_conv = 1
    i_bn = 1

    # Darknet backbone
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, 32)
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, 64, strides=2)
    for _ in range(1):
        x, i_conv, i_bn = _residual_block(x, w_darknet, w_h5, i_conv, i_bn, 32)
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, 128, strides=2)
    for _ in range(2):
        x, i_conv, i_bn = _residual_block(x, w_darknet, w_h5, i_conv, i_bn, 64)
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, 256, strides=2)
    for _ in range(8):
        x, i_conv, i_bn = _residual_block(x, w_darknet, w_h5, i_conv, i_bn, 128)
    skip_36 = x
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, 512, strides=2)
    for _ in range(8):
        x, i_conv, i_bn = _residual_block(x, w_darknet, w_h5, i_conv, i_bn, 256)
    skip_61 = x
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, 1024, strides=2)
    for _ in range(4):
        x, i_conv, i_bn = _residual_block(x, w_darknet, w_h5, i_conv, i_bn, 512)

    # Output block 1
    x, boxes_1, classes_1, i_conv, i_bn = _output_block(
        x, w_darknet, w_h5, i_conv, i_bn, 512, n_classes, _ANCHORS_1, scale=32
    )

    # Output block 2
    # Note that we don't count operations for UpSampling2D layers. This
    # is because the upsampling is nearest-neighbor, which requires no
    # arithmetic operations.
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, 256, kernel_size=1)
    x = tuple(UpSampling2D(size=2, interpolation="nearest")(x_i) for x_i in x)
    x = tuple(tf.concat(x_i, axis=-1) for x_i in zip(x, skip_61))
    x, boxes_2, classes_2, i_conv, i_bn = _output_block(
        x, w_darknet, w_h5, i_conv, i_bn, 256, n_classes, _ANCHORS_2, scale=16
    )

    # Output block 3
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, 128, kernel_size=1)
    x = tuple(UpSampling2D(size=2)(x_i) for x_i in x)
    x = tuple(tf.concat(x_i, axis=-1) for x_i in zip(x, skip_36))
    x, boxes_3, classes_3, i_conv, i_bn = _output_block(
        x, w_darknet, w_h5, i_conv, i_bn, 128, n_classes, _ANCHORS_3, scale=8
    )

    boxes = tf.concat([boxes_1, boxes_2, boxes_3], axis=1)
    classes = tf.concat([classes_1, classes_2, classes_3], axis=1)
    return EventModel(inputs=inputs, outputs=[boxes, classes])


# noinspection PyDefaultArgument
def visualize_image(
    image,
    coordinates,
    scores,
    classes,
    with_text=True,
    font_scale=1.0,
    font_thickness=1,
    image_scale=1.0,
    color=(0, 255, 255),
    rect_thickness=2,
    class_names=COCO_NAMES,
):
    y_1, x_1, y_2, x_2 = [coordinates[:, i] for i in range(4)]
    image = np.copy(visual_as_uint8(rescale_image(image, image_scale)))
    for x_1_i, x_2_i, y_1_i, y_2_i, class_i, score_i in zip(x_1, x_2, y_1, y_2, classes, scores):
        x_1_i, x_2_i, y_1_i, y_2_i = scale_coordinates((x_1_i, x_2_i, y_1_i, y_2_i), image_scale)
        cv.rectangle(image, (x_1_i, y_1_i), (x_2_i, y_2_i), color=color, thickness=rect_thickness)
        if with_text:
            text = "{} - {:.2f}%".format(class_names[class_i].capitalize(), score_i * 100.0)
            cv.putText(
                image,
                text,
                (x_1_i, y_1_i - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color,
                thickness=font_thickness,
            )
    return image


# noinspection PyDefaultArgument
def visualize_video(
    video,
    results,
    with_text=True,
    font_scale=1.0,
    font_thickness=1,
    image_scale=1.0,
    color=(0, 255, 255),
    rect_thickness=2,
    class_names=COCO_NAMES,
):
    visualized = tf.stack(
        [
            visualize_image(
                frame,
                coordinates_t,
                scores_t,
                classes_t,
                with_text=with_text,
                font_scale=font_scale,
                font_thickness=font_thickness,
                image_scale=image_scale,
                color=color,
                rect_thickness=rect_thickness,
                class_names=class_names,
            )
            for frame, (coordinates_t, scores_t, classes_t) in zip(video, results)
        ]
    )
    return tf.cast(visualized, tf.uint8)


_ANCHORS_1 = [(116, 90), (156, 198), (373, 326)]
_ANCHORS_2 = [(30, 61), (62, 45), (59, 119)]
_ANCHORS_3 = [(10, 13), (16, 30), (33, 23)]


# noinspection SpellCheckingInspection
def _box_to_int_xywh(box):
    y_1, x_1, y_2, x_2 = np.round(box).astype("int")
    w = x_2 - x_1
    h = y_2 - y_1
    return x_1, y_1, w, h


def _conv_block(
    x,
    w_darknet,
    w_h5,
    i_conv,
    i_bn,
    filters,
    kernel_size=3,
    strides=1,
    use_batch_norm=True,
    final=False,
):
    if strides == 1:
        padding = "same"
    else:
        padding = "valid"
        x = tuple([ZeroPadding2D(((1, 0), (1, 0)))(x_i) for x_i in x])
    linear = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
    linear_wrapper = LayerWrapper(linear, incremental=True, buffered=False)
    gate = Gate()
    other = BatchNormalization() if use_batch_norm else Bias()

    # Using this name allows us to find the BN layer for a given gate.
    # This is used by the apply_bn_gamma function.
    other_wrapper = LayerWrapper(
        other, incremental=False, buffered=False, name="bn_wrapper_{}".format(id(gate))
    )

    x = linear_wrapper(x)
    x = Accumulator()(x)
    x = other_wrapper(x)
    x = LayerWrapper(LeakyReLU(alpha=0.1), incremental=False, buffered=False)(x)
    if not final:
        x = gate(x)

    if w_darknet is not None:
        other_wrapper.build(other_wrapper.input_shape)
        if use_batch_norm:
            # Darknet uses order [beta, gamma, mean, variance].
            w_bn = np.fromfile(w_darknet, dtype=np.float32, count=4 * filters)
            w_bn = w_bn.reshape((4, filters))[[1, 0, 2, 3]]
            other.set_weights(w_bn)
        else:
            w_bias = np.fromfile(w_darknet, dtype=np.float32, count=filters)
            w_bias = w_bias.reshape(other.bias.shape)
            other.set_weights([w_bias])

        # A Darknet kernel has shape (c_out, c_in, h, w).
        linear_wrapper.build(linear_wrapper.input_shape)
        shape_tf = linear.kernel.shape
        shape_darknet = (shape_tf[3], shape_tf[2], shape_tf[0], shape_tf[1])
        w_linear = np.fromfile(w_darknet, dtype=np.float32, count=int(np.prod(shape_darknet)))
        w_linear = w_linear.reshape(shape_darknet).transpose([2, 3, 1, 0])
        linear.set_weights([w_linear])

    if w_h5 is not None:
        other_wrapper.build(other_wrapper.input_shape)
        if use_batch_norm:
            key = "batch_normalization_{}".format(i_bn)
            w_bn = w_h5[key][key]
            other.set_weights(
                [w_bn["gamma:0"], w_bn["beta:0"], w_bn["moving_mean:0"], w_bn["moving_variance:0"]]
            )
            i_bn += 1
        else:
            key = "conv2d_{}".format(i_conv)
            other.set_weights([np.array(w_h5[key][key]["bias:0"]).reshape(other.bias.shape)])
        linear_wrapper.build(linear_wrapper.input_shape)
        key = "conv2d_{}".format(i_conv)
        linear.set_weights([w_h5[key][key]["kernel:0"]])

    return x, i_conv + 1, i_bn


def _output_block(x, w_darknet, w_h5, i_conv, i_bn, filters, n_classes, anchors, scale):
    for _ in range(2):
        x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, filters, kernel_size=1)
        x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, filters * 2)
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, filters, kernel_size=1)
    skip = x
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, filters * 2)
    x, i_conv, i_bn = _conv_block(
        x,
        w_darknet,
        w_h5,
        i_conv,
        i_bn,
        len(anchors) * (n_classes + 5),
        kernel_size=1,
        use_batch_norm=False,
        final=True,
    )
    x = Unmask()(x)

    boxes = []
    classes = []
    for i, anchor in enumerate(anchors):
        offset = i * (n_classes + 5)
        grid_x, grid_y = tf.meshgrid(
            tf.range(tf.shape(x)[2], dtype=x.dtype), tf.range(tf.shape(x)[1], dtype=x.dtype)
        )
        box_x = scale * (tf.sigmoid(x[..., offset + 0]) + grid_x)
        box_y = scale * (tf.sigmoid(x[..., offset + 1]) + grid_y)
        box_w = tf.exp(x[..., offset + 2]) * anchor[0]
        box_h = tf.exp(x[..., offset + 3]) * anchor[1]
        class_scores = tf.sigmoid(x[..., offset + 5 : offset + 5 + n_classes])
        box_score = tf.sigmoid(x[..., offset + 4]) * tf.reduce_max(class_scores, axis=-1)

        box_x_1 = box_x - box_w / 2
        box_x_2 = box_x + box_w / 2
        box_y_1 = box_y - box_h / 2
        box_y_2 = box_y + box_h / 2
        boxes_i = tf.stack([box_y_1, box_x_1, box_y_2, box_x_2, box_score], axis=-1)
        boxes_i = tf.reshape(boxes_i, (tf.shape(boxes_i)[0], -1, boxes_i.shape[-1]))
        boxes.append(boxes_i)

        classes_i = tf.argmax(class_scores, axis=-1)
        classes_i = tf.reshape(classes_i, (tf.shape(classes_i)[0], -1))
        classes.append(classes_i)

    boxes = tf.concat(boxes, axis=1)
    classes = tf.concat(classes, axis=1)
    return skip, boxes, classes, i_conv, i_bn


def _residual_block(x, w_darknet, w_h5, i_conv, i_bn, filters):
    skip = x
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, filters, kernel_size=1)
    x, i_conv, i_bn = _conv_block(x, w_darknet, w_h5, i_conv, i_bn, filters * 2)
    x = tuple(tf.stack(x_i, axis=1) for x_i in zip(skip, x))
    x = LayerWrapper(Fuse(), incremental=True, buffered=False)(x)
    return x, i_conv, i_bn

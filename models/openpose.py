import cv2 as cv
import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from tensorflow.keras.layers import *

from datasets.jhmdb import MPII_REMAP_INDICES
from eventnn.layers import Accumulator, Bias, Gate, LayerWrapper, Mask, Unmask
from eventnn.model import EventModel
from utils.misc import (
    rescale_image,
    resize_image,
    scale_coordinates,
    visual_as_float,
    visual_as_uint8,
)

# Resources consulted:
# https://openaccess.thecvf.com/content_cvpr_2017/html/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.html
# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/python/demo.ipynb


# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/src/connect56LineVec.m
COCO_LIMB_PAF_CHANNELS = [
    [12, 13],
    [20, 21],
    [14, 15],
    [16, 17],
    [22, 23],
    [24, 25],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [28, 29],
    [30, 31],
    [34, 35],
    [32, 33],
    [36, 37],
    [18, 19],
    [26, 27],
]

# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/src/connect56LineVec.m
COCO_LIMB_TYPES = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
    [2, 16],
    [5, 17],
]

# The names of select Gate layers, used for inspection and visualization.
GATE_NAMES = [
    "vgg_pool_1",
    "vgg_pool_2",
    "vgg_pool_3",
    "vgg_output",
    "block_2_heatmap",
    "block_3_heatmap",
    "block_4_heatmap",
    "block_5_heatmap",
]


# Maps the MPII joint order to the order expected by OpenPose.
# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/evalMPII.m
MPII_JOINT_ORDER = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, 6, 7]

# Inverse of the above. Maps the output of OpenPose to the MPII order.
# [MPII_JOINT_ORDER.index(i) for i in range(len(MPII_JOINT_ORDER))]
MPII_JOINT_ORDER_INV = [10, 9, 8, 11, 12, 13, 14, 15, 1, 0, 4, 3, 2, 5, 6, 7]

# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/src/connect43LineVec.m
MPII_LIMB_PAF_CHANNELS = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [12, 13],
    [14, 15],
    [22, 23],
    [24, 25],
    [26, 27],
    [16, 17],
    [18, 19],
    [20, 21],
]

# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/src/connect43LineVec.m
MPII_LIMB_TYPES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 14],
    [14, 11],
    [11, 12],
    [12, 13],
    [14, 8],
    [8, 9],
    [9, 10],
]


def openpose_coco(input_size, batch_size=1, npz_weights=None, delta_based=True):
    return _openpose(
        input_size, batch_size, npz_weights, delta_based, n_joint_types=19, n_joint_pairs=38
    )


def openpose_mpii(input_size, batch_size=1, npz_weights=None, delta_based=True):
    return _openpose(
        input_size, batch_size, npz_weights, delta_based, n_joint_types=16, n_joint_pairs=28
    )


# The implementation of this function was informed by the original
# authors' code at https://bit.ly/3h95d6U.
def postprocess_image(image, paf, heatmap, mode="mpii"):
    paf = np.array(resize_image(paf, image.shape[0:2], method="bicubic"))
    heatmap = np.array(resize_image(heatmap, image.shape[0:2], method="bicubic"))
    n_joint_types = heatmap.shape[-1]
    if mode == "mpii":
        limb_paf_channels = MPII_LIMB_PAF_CHANNELS
        limb_types = MPII_LIMB_TYPES
    elif mode == "coco":
        limb_paf_channels = COCO_LIMB_PAF_CHANNELS
        limb_types = COCO_LIMB_TYPES
    else:
        raise ValueError("Invalid mode {}.".format(mode))
    n_limb_types = len(limb_types)

    # Extract local peaks (joints) from the heatmaps.
    all_joints = []
    joint_id = 0
    for j in range(n_joint_types):
        heatmap_i = gaussian_filter(heatmap[..., j], sigma=3)
        joints = _find_peaks(heatmap_i)
        joints_type_i = []
        for joint in joints:
            x, y = joint[1], joint[0]
            score = heatmap_i[y, x]
            joints_type_i.append((x, y, score, joint_id))
            joint_id += 1
        all_joints.append(joints_type_i)

    # Connect joints into limbs.
    all_limbs = []
    for i in range(n_limb_types):
        paf_vectors = paf[:, :, limb_paf_channels[i]]
        joints_type_1 = all_joints[limb_types[i][0]]
        joints_type_2 = all_joints[limb_types[i][1]]

        limb_candidates = []
        for j, peak_1 in enumerate(joints_type_1):
            for k, peak_2 in enumerate(joints_type_2):
                limb_vector = np.array(peak_2[:2]) - np.array(peak_1[:2])
                limb_length = np.linalg.norm(limb_vector)
                if limb_length == 0:
                    continue

                # Estimate a line integral over the PAF.
                point_scores = []
                x, y = peak_1[:2]
                dx, dy = limb_vector / (_N_INTEGRAL - 1)
                for _ in range(_N_INTEGRAL):
                    point_scores.append(
                        paf_vectors[round(y), round(x)] @ (limb_vector / limb_length)
                    )
                    x += dx
                    y += dy
                point_scores = np.array(point_scores)

                # The score incorporates a distance prior.
                prior = min(0.5 * image.shape[0] / limb_length - 1, 0)
                score = np.mean(point_scores) + prior
                criterion_1 = np.mean(point_scores > _SCORE_THRESHOLD) > _MIN_FRACTION
                criterion_2 = score > 0
                if criterion_1 and criterion_2:
                    limb_candidates.append([j, k, score])

        # Greedily choose the limb candidates with the highest scores.
        limb_candidates.sort(key=lambda a: a[2], reverse=True)
        j_used = []
        k_used = []
        limbs_type_i = []
        for j, k, _ in limb_candidates:
            if j not in j_used and k not in k_used:
                j_used.append(j)
                k_used.append(k)
                id_1 = joints_type_1[j][3]
                id_2 = joints_type_2[k][3]
                limbs_type_i.append((id_1, id_2))
        all_limbs.append(limbs_type_i)

    # Connect limbs into people.
    people = np.empty((0, n_joint_types), dtype=int)
    for i, limbs_type_i in enumerate(all_limbs):
        index_a, index_b = limb_types[i]

        for limb in limbs_type_i:
            joint_id_1, joint_id_2 = limb

            matches = []
            for j, person in enumerate(people):
                if person[index_a] == limb[0] or person[index_b] == limb[1]:
                    matches.append(j)
                    people[j][index_a] = joint_id_1
                    people[j][index_b] = joint_id_2

            # If a limb got paired to two different people, merge those
            # people together.
            if len(matches) == 2:
                j_1, j_2 = matches
                merge_indices = people[j_1] == -1
                people[j_1][merge_indices] = people[j_2][merge_indices]
                people = np.delete(people, j_2, axis=0)

            # If the limb didn't get paired to anyone, create a new
            # person.
            elif len(matches) == 0:
                row = np.full((n_joint_types,), -1, dtype=int)
                row[index_a] = joint_id_1
                row[index_b] = joint_id_2
                people = np.vstack([people, row])

    joints_flattened = [joint for joints_i in all_joints for joint in joints_i]
    output = []
    for i, person in enumerate(people):
        # Remove people with only a few joints.
        # Note that the original implementation (https://bit.ly/3h95d6U)
        # also filters people based on total score. However, because
        # this is not mentioned in the text of the main paper (and for
        # simplicity) we do not apply score filtering.
        if np.count_nonzero(person >= 0) < _MIN_JOINTS:
            continue

        # Look up the coordinates of the joints.
        person_coordinates = np.full((n_joint_types, 2), -1, dtype=int)
        for j, joint_id in enumerate(person):
            if joint_id != -1:
                person_coordinates[j] = joints_flattened[joint_id][:2]
        output.append(person_coordinates)

    return np.array(output)


def postprocess_pck_jhmdb_mpii(y_true, y_pred):
    frames, joints = y_true
    pafs, heatmaps = y_pred
    people_true_batch = []
    people_pred_batch = []
    for i in range(frames.shape[0]):
        person_true = np.array(joints[i])
        people_pred = postprocess_image(frames[i], pafs[i], heatmaps[i], mode="mpii")
        if len(people_pred) == 0:
            person_pred = np.full(person_true.shape, -1, dtype=int)
        else:
            # Choose the predicted person with the most joints.
            people_pred = people_pred[:, MPII_JOINT_ORDER_INV][:, MPII_REMAP_INDICES]
            n_best = 0
            person_pred = people_pred[0]
            for person in people_pred:
                # Select just the first coordinate (both should be
                # negative if no joint was detected).
                n = np.count_nonzero(person[:, 0] >= 0)
                if n > n_best:
                    n_best = n
                    person_pred = person
        people_true_batch.append(person_true)
        people_pred_batch.append(person_pred)
    return people_true_batch, people_pred_batch


def postprocess_pck_mpii(y_true, y_pred):
    frames, joints, n_people = y_true
    pafs, heatmaps = y_pred
    people_true_batch = []
    people_pred_batch = []
    for i in range(frames.shape[0]):
        people_true_batch.append(list(np.array(joints[i][: n_people[i]])))
        people_pred_batch.append(
            list(postprocess_image(frames[i], pafs[i], heatmaps[i], mode="mpii"))
        )
    return people_true_batch, people_pred_batch


def postprocess_pckh_mpii(y_true, y_pred):
    frames, frame_masks, joints, heads = y_true
    pafs, heatmaps = y_pred
    people_true_batch = []
    people_pred_batch = []
    for i in range(frames.shape[0]):
        if not bool(frame_masks[i]):
            continue
        people_true_batch.append(list(np.array(joints[i])))
        people_pred_batch.append(
            list(postprocess_image(frames[i], pafs[i], heatmaps[i], mode="mpii"))
        )
    return people_true_batch, people_pred_batch


def postprocess_video(video, pafs, heatmaps, mode="mpii"):
    return [
        postprocess_image(frame, pafs_t, heatmaps_t, mode=mode)
        for frame, pafs_t, heatmaps_t in zip(video, pafs, heatmaps)
    ]


def preprocess_image(image):
    # The model expects BGR input.
    image = visual_as_float(image)
    image = tf.reverse(image, axis=[-1])
    image = image * 255.0 / 256.0 - 0.5
    return image


def preprocess_video(video):
    return tf.stack([preprocess_image(frame) for frame in video])


def undo_preprocess_image(image):
    image = (image + 0.5) * 256.0 / 255.0
    image = tf.reverse(image, axis=[-1])
    return image


def undo_preprocess_video(video):
    return tf.stack([undo_preprocess_image(frame) for frame in video])


def visualize_image(
    image, output, image_scale=1.0, mode="mpii", line_thickness=2, circle_radius=4
):
    image = np.copy(visual_as_uint8(rescale_image(image, image_scale)))
    if mode == "mpii":
        displayed_limb_types = _MPII_DISPLAYED_LIMB_TYPES
    elif mode == "coco":
        displayed_limb_types = _COCO_DISPLAYED_LIMB_TYPES
    else:
        raise ValueError("Invalid mode {}.".format(mode))
    for limb_type in displayed_limb_types:
        for person in output:
            j_1, j_2 = limb_type
            joint_1, joint_2 = person[limb_type]
            if -1 in joint_1 or -1 in joint_2:
                continue
            joint_1 = scale_coordinates(joint_1, image_scale)
            joint_2 = scale_coordinates(joint_2, image_scale)
            color_1 = tuple(reversed(_COLORS[j_1]))
            color_2 = tuple(reversed(_COLORS[j_2]))
            cv.line(image, joint_1, joint_2, color=color_2, thickness=line_thickness)
            cv.circle(image, joint_1, circle_radius, color=color_1, thickness=-1)
            cv.circle(image, joint_2, circle_radius, color=color_2, thickness=-1)
    return visual_as_float(image)


def visualize_video(video, output, image_scale=1.0, mode="mpii"):
    return tf.stack(
        [
            visualize_image(frame, output_t, image_scale=image_scale, mode=mode)
            for frame, output_t in zip(video, output)
        ]
    )


# Limb types to display for COCO
_COCO_DISPLAYED_LIMB_TYPES = COCO_LIMB_TYPES[:-2]

# Colors for visualization
_COLORS = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]


# Amount by which heatmaps should be blurred
_GAUSSIAN_SIGMA = 3

# Minimum fraction of points along the PAF integral that must match (see
# also _SCORE_THRESHOLD)
_MIN_FRACTION = 0.8

# Minimum number of joints for a detected person
_MIN_JOINTS = 4

# Limb types to display for MPII
_MPII_DISPLAYED_LIMB_TYPES = MPII_LIMB_TYPES

# Number of points to integrate between joints
_N_INTEGRAL = 10

# Number of two-branch stages in the model
_N_STAGES = 5

# Threshold for detecting a peak in the heatmap
_PEAK_THRESHOLD = 0.1

# Minimum score for counting a point along the PAF integral as a match
# (see also _MIN_FRACTION)
_SCORE_THRESHOLD = 0.05


def _conv_block(
    x,
    w,
    weight_name,
    delta_based,
    filters,
    kernel_size,
    use_relu=True,
    final=False,
    gate_name=None,
):
    linear = Conv2D(filters, kernel_size, padding="same", use_bias=False)
    linear_wrapper = LayerWrapper(linear, incremental=delta_based, buffered=(not delta_based))
    bias = Bias()
    bias_wrapper = LayerWrapper(bias, incremental=False, buffered=False)

    x = linear_wrapper(x)
    if delta_based:
        x = Accumulator()(x)
    x = bias_wrapper(x)
    if use_relu:
        x = LayerWrapper(ReLU(), incremental=False, buffered=False)(x)
    if not final:
        x = Gate(delta_based=delta_based, name=gate_name)(x)

    if w is not None:
        # A Caffe kernel has shape (c_out, c_in, h, w).
        linear_wrapper.build(linear_wrapper.input_shape)
        w_linear = w["{}-0".format(weight_name)]
        w_linear = w_linear.transpose([2, 3, 1, 0])
        linear.set_weights([w_linear])

        bias_wrapper.build(bias_wrapper.input_shape)
        w_bias = w["{}-1".format(weight_name)]
        w_bias = w_bias.reshape(bias.bias.shape)
        bias.set_weights([w_bias])

    return x


def _openpose(input_size, batch_size, npz_weights, delta_based, n_joint_types, n_joint_pairs):
    inputs = Input(batch_input_shape=(batch_size,) + input_size + (3,))
    x = inputs

    if npz_weights is not None:
        w = np.load(npz_weights)
    else:
        w = None

    # Model input
    x = Mask()(x)
    x = Gate(delta_based=delta_based)(x)

    # VGG backbone
    x = _vgg_conv_stack(x, w, "conv1", delta_based, n=2, filters=64)
    x = _vgg_pool_block(x, delta_based, gate_name="vgg_pool_1")
    x = _vgg_conv_stack(x, w, "conv2", delta_based, n=2, filters=128)
    x = _vgg_pool_block(x, delta_based, gate_name="vgg_pool_2")
    x = _vgg_conv_stack(x, w, "conv3", delta_based, n=4, filters=256)
    x = _vgg_pool_block(x, delta_based, gate_name="vgg_pool_3")
    x = _vgg_conv_stack(x, w, "conv4", delta_based, n=2, filters=512)
    x = Gate(delta_based=delta_based, name="vgg_output")(x)

    # CPM layers
    x = _conv_block(x, w, "conv4_3_CPM", delta_based, filters=256, kernel_size=3)
    x = _conv_block(x, w, "conv4_4_CPM", delta_based, filters=128, kernel_size=3)
    cpm_skip = x
    x_1 = x
    x_2 = x
    for j in range(1, 4):
        x_1 = _conv_block(
            x_1, w, "conv5_{}_CPM_L1".format(j), delta_based, filters=128, kernel_size=3
        )
        x_2 = _conv_block(
            x_2, w, "conv5_{}_CPM_L2".format(j), delta_based, filters=128, kernel_size=3
        )
    x_1 = _conv_block(x_1, w, "conv5_4_CPM_L1", delta_based, filters=512, kernel_size=1)
    x_2 = _conv_block(x_2, w, "conv5_4_CPM_L2", delta_based, filters=512, kernel_size=1)
    x_1 = _conv_block(
        x_1, w, "conv5_5_CPM_L1", delta_based, filters=n_joint_pairs, kernel_size=1, use_relu=False
    )
    x_2 = _conv_block(
        x_2, w, "conv5_5_CPM_L2", delta_based, filters=n_joint_types, kernel_size=1, use_relu=False
    )

    # Two-branch stages
    for i in range(2, 2 + _N_STAGES):
        x = tuple(tf.concat(x_i, axis=-1) for x_i in zip(x_1, x_2, cpm_skip))
        x_1 = x
        x_2 = x
        for j in range(1, 6):
            x_1 = _conv_block(
                x_1, w, "Mconv{}_stage{}_L1".format(j, i), delta_based, filters=128, kernel_size=7
            )
            x_2 = _conv_block(
                x_2, w, "Mconv{}_stage{}_L2".format(j, i), delta_based, filters=128, kernel_size=7
            )
        x_1 = _conv_block(
            x_1, w, "Mconv6_stage{}_L1".format(i), delta_based, filters=128, kernel_size=1
        )
        x_2 = _conv_block(
            x_2, w, "Mconv6_stage{}_L2".format(i), delta_based, filters=128, kernel_size=1
        )
        x_1 = _conv_block(
            x_1,
            w,
            "Mconv7_stage{}_L1".format(i),
            delta_based,
            filters=n_joint_pairs,
            kernel_size=1,
            use_relu=False,
            final=(i == _N_STAGES + 1),
        )
        x_2 = _conv_block(
            x_2,
            w,
            "Mconv7_stage{}_L2".format(i),
            delta_based,
            filters=n_joint_types,
            kernel_size=1,
            use_relu=False,
            final=(i == _N_STAGES + 1),
            gate_name="block_{}_heatmap".format(i),
        )

    # Model output
    x_1 = Unmask()(x_1)
    x_2 = Unmask()(x_2)

    return EventModel(inputs=inputs, outputs=[x_1, x_2])


# scikit-image has a function that does this, but it's *very* slow.
def _find_peaks(array):
    shifted_left = np.zeros_like(array)
    shifted_left[:, :-1] = array[:, 1:]
    shifted_right = np.zeros_like(array)
    shifted_right[:, 1:] = array[:, :-1]
    shifted_up = np.zeros_like(array)
    shifted_up[:-1, :] = array[1:, :]
    shifted_down = np.zeros_like(array)
    shifted_down[1:, :] = array[:-1, :]
    peaks = np.logical_and.reduce(
        (
            array >= shifted_left,
            array >= shifted_right,
            array >= shifted_up,
            array >= shifted_down,
            array > _PEAK_THRESHOLD,
        )
    )
    return zip(*np.nonzero(peaks))


def _vgg_conv_stack(x, w, name_base, delta_based, n, filters):
    for i in range(n):
        weight_name = "{}_{}".format(name_base, i + 1)
        x = _conv_block(x, w, weight_name, delta_based, filters, kernel_size=3, final=(i == n - 1))
    return x


def _vgg_pool_block(x, delta_based, gate_name=None):
    x = LayerWrapper(MaxPooling2D(pool_size=2), incremental=False, buffered=True)(x)
    x = Gate(delta_based=delta_based, name=gate_name)(x)
    return x

#!/usr/bin/env python3

import os
import os.path as path
import shutil
from argparse import ArgumentParser

import tensorflow as tf

from datasets.sintel import read_flo
from utils.misc import find_unused_dirname


def main(args):
    tmp_dir = find_unused_dirname("tmp")
    os.makedirs(tmp_dir)

    try:
        shutil.unpack_archive(args.zip_file, tmp_dir)

        train_in_dir = path.join(tmp_dir, "training")
        test_in_dir = path.join(tmp_dir, "test")
        train_out_dir = path.join(args.out_dir, "train")
        test_out_dir = path.join(args.out_dir, "test")
        os.makedirs(train_out_dir)
        os.makedirs(test_out_dir)

        # Move things that don't need modification.
        shutil.move(path.join(train_in_dir, "final"), path.join(train_out_dir, "frames"))
        shutil.move(path.join(train_in_dir, "flow_viz"), path.join(train_out_dir, "flow_vis"))
        shutil.move(path.join(train_in_dir, "occlusions"), path.join(train_out_dir, "occlusions"))
        shutil.move(path.join(test_in_dir, "final"), path.join(test_out_dir, "frames"))

        # Convert .flo files to serialized tensors.
        flow_in_dir = path.join(train_in_dir, "flow")
        flow_out_dir = path.join(train_out_dir, "flow")
        os.makedirs(flow_out_dir)
        for video_dir in os.listdir(flow_in_dir):
            video_in_dir = path.join(flow_in_dir, video_dir)
            video_out_dir = path.join(flow_out_dir, video_dir)
            for flow_filename in os.listdir(video_in_dir):
                flow = read_flo(path.join(video_in_dir, flow_filename))
                proto_filename = "{}.proto".format(path.splitext(flow_filename)[0])
                tf.io.write_file(
                    path.join(video_out_dir, proto_filename), tf.io.serialize_tensor(flow)
                )

    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(tmp_dir)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("zip_file", help="zip file containing the dataset")
    parser.add_argument(
        "out_dir", help="top-level directory where the processed output should be saved"
    )

    # Optional arguments

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

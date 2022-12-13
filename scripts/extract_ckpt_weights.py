#!/usr/bin/env python3

from argparse import ArgumentParser

import numpy as np
import tensorflow as tf


def main(args):
    reader = tf.train.load_checkpoint(args.ckpt_dir)
    keys = reader.get_variable_to_shape_map().keys()
    output = {}
    for key in keys:
        output[key] = reader.get_tensor(key)
    np.savez(args.output_file, **output)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("ckpt_dir", help="checkpoint directory containing the model")
    parser.add_argument("output_file", help="location to save the output weights (a .npz file)")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

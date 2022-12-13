#!/usr/bin/env python3

from argparse import ArgumentParser

# Install caffe via "sudo apt install caffe-cpu"
import caffe
import numpy as np


def main(args):
    model = caffe.Net(args.prototxt_file, args.caffemodel_file, caffe.TEST)
    output = {}
    for key, blob_vec in model.params.items():
        for i, blob in enumerate(blob_vec):
            output["{}-{}".format(key, i)] = blob.data
    np.savez(args.output_file, **output)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("prototxt_file", help="prototxt file containing the model configuration")
    parser.add_argument("caffemodel_file", help="caffemodel file containing the model parameters")
    parser.add_argument("output_file", help="location to save the output weights (a .npz file)")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

#!/usr/bin/env python3

from argparse import ArgumentParser

import numpy as np
# Install torch via the usual methods (e.g., conda).
import torch


def main(args):
    output = torch.load_data(args.pth_tar_file, map_location=torch.device("cpu"))
    np.savez(args.output_file, **dict(output))


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("pth_tar_file", help="pth.tar file containing the model parameters")
    parser.add_argument("output_file", help="location to save the output weights (a .npz file)")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

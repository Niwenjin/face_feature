import os
import cv2
import numpy as np
from ijb_evals import Caffe_model_interf


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model_file",
        type=str,
        default=None,
        help="Saved model, keras h5 / pytorch jit pth / onnx / mxnet / rknn / caffe prototxt",
    )
    args = parser.parse_known_args(argv)[0]

    return args


if __name__ == "__main__":
    import sys

    args = parse_arguments(sys.argv[1:])

    interf_func = Caffe_model_interf(args.model_file)

    embedding_1 = np.array(interf_func(cv2.imread("1.jpg")))
    embedding_1 = embedding_1 / np.linalg.norm(embedding_1)
    embedding_2 = np.array(interf_func(cv2.imread("2.jpg")))
    embedding_2 = embedding_2 / np.linalg.norm(embedding_2)

    print(f"Embedding 1: {embedding_1[0]},\nEmbedding 2: {embedding_2[0]}")

    score_func = lambda feat1, feat2: 1 - np.linalg.norm(feat1 - feat2, axis=-1) / 3.451
    score = score_func(embedding_1, embedding_2)
    print(f"Score: {score[0]}")

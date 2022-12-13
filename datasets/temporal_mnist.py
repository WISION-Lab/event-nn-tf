import numpy as np
from tensorflow.keras.utils import to_categorical


def load_data(npz_file, split):
    data = np.load(npz_file)
    if split == "train":
        return (
            np.expand_dims(data["x_train"], axis=0),
            np.expand_dims(to_categorical(data["y_train"]), axis=0),
        )
    else:
        return (
            np.expand_dims(data["x_test"], axis=0),
            np.expand_dims(to_categorical(data["y_test"]), axis=0),
        )

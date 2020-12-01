"""Module for building plots"""

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2


def plot_true_pred_flatten_images(true_pred_pairs: List[Tuple[np.ndarray, np.ndarray]]):
    flatten_length = true_pred_pairs[0][0].shape[0]
    original_image_size = (int(np.sqrt(flatten_length)), int(np.sqrt(flatten_length)))

    fig, axs = plt.subplots(len(true_pred_pairs), 2)

    axs[0][0].set_title('Original binary')
    axs[0][1].set_title('Predicted binary')

    for idx, (curr_true_flatten, curr_pred_flatten) in enumerate(true_pred_pairs):
        curr_true_image = np.reshape(curr_true_flatten,
                                     newshape=original_image_size)
        curr_pred_image = np.reshape(curr_pred_flatten,
                                     newshape=original_image_size)

        axs[idx][0].imshow(curr_true_image)
        axs[idx][1].imshow(curr_pred_image)

    plt.tight_layout()
    plt.show()

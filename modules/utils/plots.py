"""Module for building plots"""

from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2


def plot_images_compare(data: Dict[str, List[np.ndarray]]):
    """
    Creates plot with images.
    Where rows are used for the same class image.
    And columns for their type (result, prediction, original)

    Example:

    >> data = {
    >>    'original_image': flatten_images_original,  # List[np.ndarray]
    >>    'prediction_image': predictions  # List[np.ndarray]
    >> }

    :param data: dictionary with images to draw
    """

    keys = list(data.keys())

    values_from_dict = data[keys[0]]
    image_side_length = int(
        np.sqrt(values_from_dict[0].shape[0])
    )

    original_image_size = (image_side_length, image_side_length)

    fig, axs = plt.subplots(len(values_from_dict), len(keys))

    for idx, curr_key in enumerate(keys):
        axs[0][idx].set_title(curr_key)

    for idx, row_values in enumerate(zip(*data.values())):

        for idx_of_row_image, curr_image_flatten in enumerate(row_values):
            curr_image = np.reshape(curr_image_flatten,
                                    newshape=original_image_size)

            axs[idx][idx_of_row_image].imshow(curr_image)

    plt.tight_layout()
    plt.show()

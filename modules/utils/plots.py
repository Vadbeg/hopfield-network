"""Module for building plots"""

from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2


def plot_true_pred_flatten_images(data: Dict[str, List[np.ndarray]]):
    flatten_images_original_list = data['original_image']
    flatten_images_noise_list = data['noise_image']
    flatten_predictions_list = data['prediction_image']

    flatten_length_side = flatten_images_original_list[0].shape[0]
    original_image_size = (int(np.sqrt(flatten_length_side)),
                           int(np.sqrt(flatten_length_side)))

    fig, axs = plt.subplots(len(flatten_images_original_list), 3)

    axs[0][0].set_title('Original binary')
    axs[0][1].set_title('Noise binary')
    axs[0][2].set_title('Predictor binary')

    data = zip(flatten_images_original_list,
               flatten_images_noise_list,
               flatten_predictions_list)

    for idx, (curr_original, curr_noise, curr_prediction) in enumerate(data):

        curr_original_image = np.reshape(curr_original,
                                         newshape=original_image_size)
        curr_noise_image = np.reshape(curr_noise,
                                      newshape=original_image_size)
        curr_pred_image = np.reshape(curr_prediction,
                                     newshape=original_image_size)

        axs[idx][0].imshow(curr_original_image)
        axs[idx][1].imshow(curr_noise_image)
        axs[idx][2].imshow(curr_pred_image)

    plt.tight_layout()
    plt.show()

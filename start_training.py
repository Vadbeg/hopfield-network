"""Module for training starting"""

import glob
from typing import List, Tuple, Dict

import numpy as np

from modules.data.dataset import Dataset
from modules.network.hopfield import HopfieldNetwork
from modules.utils.plots import plot_images_compare
from modules.train import get_trained_model
from config import Config


def evaluation(model: HopfieldNetwork,
               image_paths: List[str],
               image_size: Tuple[int, int] = (256, 256),
               num_iter: int = 20,
               threshold: int = 50,
               add_noise: bool = False) -> Dict[str, List[np.ndarray]]:
    """
    Evaluates HopfieldNetwork model on images from image_paths

    :param model: model instance
    :param image_paths: paths to the images
    :param image_size: image size
    :param num_iter: images passes threw network num_iter times
    :param threshold: threshold for sign function in network
    :param add_noise: if True adds noise to input image
    :return:
    """

    dataset_original = Dataset(list_of_paths=image_paths,
                               image_size=image_size,
                               add_noise=add_noise)
    flatten_images_original = dataset_original.get_all_flatten_images()

    predictions = model.predict(data=flatten_images_original)

    data = {
        'original_image': flatten_images_original,
        'prediction_image': predictions
    }

    return data


if __name__ == '__main__':
    use_numbers_dataset = False

    if use_numbers_dataset:

        image_paths_train = glob.glob(pathname='images_diff/train/*.*', recursive=True)
        image_paths_test = glob.glob(pathname='images_diff/test/*.*', recursive=True)

        model = get_trained_model(image_paths=image_paths_train,
                                  image_size=Config.image_size,
                                  asynchronous=Config.asynchronous)

        data = evaluation(model=model,
                          image_paths=image_paths_test,
                          image_size=Config.image_size,
                          num_iter=Config.num_iter,
                          threshold=Config.threshold)

        plot_images_compare(data=data)
    else:
        image_paths = glob.glob(pathname='images_same/*.*', recursive=True)

        model = get_trained_model(image_paths=image_paths,
                                  image_size=Config.image_size,
                                  asynchronous=False)

        data = evaluation(model=model,
                          image_paths=image_paths,
                          image_size=Config.image_size,
                          num_iter=Config.num_iter,
                          threshold=Config.threshold,
                          add_noise=True)

        plot_images_compare(data=data)

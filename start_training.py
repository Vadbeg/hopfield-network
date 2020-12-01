"""Module for training starting"""

import glob
from typing import List, Tuple

from modules.data.dataset import Dataset
from modules.network.hopfield import HopfieldNetwork
from modules.utils.plots import plot_true_pred_flatten_images


def get_trained_model(image_paths: List[str],
                      image_size: Tuple[int, int] = (256, 256)) -> HopfieldNetwork:
    dataset = Dataset(list_of_paths=image_paths, image_size=image_size)
    flatten_images = dataset.get_all_flatten_images()

    model = HopfieldNetwork(train_data=flatten_images, asynchronous=True)
    model.train()

    return model


def evaluation(model: HopfieldNetwork, image_paths: List[str]):
    dataset = Dataset(list_of_paths=image_paths,
                      image_size=image_size,
                      add_noise=True)
    flatten_images = dataset.get_all_flatten_images()

    predictions = model.predict(data=flatten_images, num_iter=10)

    true_prediction_pairs = list(zip(flatten_images, predictions))

    return true_prediction_pairs


if __name__ == '__main__':
    image_paths = glob.glob(pathname='images/*.*', recursive=True)

    image_size = (128, 128)

    model = get_trained_model(image_paths=image_paths,
                              image_size=image_size)

    true_prediction_pairs = evaluation(model=model, image_paths=image_paths)

    plot_true_pred_flatten_images(true_pred_pairs=true_prediction_pairs)

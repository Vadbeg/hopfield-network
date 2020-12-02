"""Module for training starting"""

import glob
from typing import List, Tuple

from modules.data.dataset import Dataset
from modules.network.hopfield import HopfieldNetwork
from modules.utils.plots import plot_true_pred_flatten_images


def get_trained_model(image_paths: List[str],
                      image_size: Tuple[int, int] = (256, 256),
                      asynchronous: bool = True) -> HopfieldNetwork:
    dataset = Dataset(list_of_paths=image_paths, image_size=image_size)
    flatten_images = dataset.get_all_flatten_images()

    model = HopfieldNetwork(train_data=flatten_images, asynchronous=asynchronous)
    model.train()

    return model


def evaluation(model: HopfieldNetwork, image_paths: List[str]):
    dataset_noise = Dataset(list_of_paths=image_paths,
                            image_size=image_size,
                            add_noise=True)
    flatten_images_noise = dataset_noise.get_all_flatten_images()

    dataset_original = Dataset(list_of_paths=image_paths,
                               image_size=image_size,
                               add_noise=False)
    flatten_images_original = dataset_original.get_all_flatten_images()

    predictions = model.predict(data=flatten_images_noise, num_iter=100, threshold=50)

    data = {
        'original_image': flatten_images_original,
        'noise_image': flatten_images_noise,
        'prediction_image': predictions
    }

    return data


if __name__ == '__main__':
    image_paths = glob.glob(pathname='images/*.*', recursive=True)[:2]

    image_size = (64, 64)

    model = get_trained_model(image_paths=image_paths,
                              image_size=image_size,
                              asynchronous=True)

    data = evaluation(model=model, image_paths=image_paths)

    plot_true_pred_flatten_images(data=data)

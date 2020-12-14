"""Module with train methods"""

from typing import List, Tuple

from modules.data.dataset import Dataset
from modules.network.hopfield import HopfieldNetwork


def get_trained_model(image_paths: List[str],
                      image_size: Tuple[int, int] = (256, 256),
                      asynchronous: bool = True,
                      projections: bool = True) -> HopfieldNetwork:
    """
    Creates and train model with data which we've get from image_paths

    :param image_paths: paths to train images
    :param image_size: size of the image
    :param asynchronous: if True uses async algorithm, else sync
    :param projections: if True uses projections algorithm, else Hebbs
    :return: model instance
    """

    dataset = Dataset(list_of_paths=image_paths, image_size=image_size)
    flatten_images = dataset.get_all_flatten_images()

    model = HopfieldNetwork(train_data=flatten_images, asynchronous=asynchronous, projections=projections)
    model.train()

    return model

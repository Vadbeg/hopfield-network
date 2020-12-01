"""Module with dataset for network"""

from typing import List, Tuple

import numpy as np
from cv2 import cv2

from modules.data.base_dataset import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, list_of_paths: List[str], image_size: Tuple[int, int] = (32, 32)):
        """
        Dataset class

        :param list_of_paths: list paths to images
        :param image_size: image size
        """

        self.list_of_paths = list_of_paths

        self.image_size = image_size

    def __preprocess_image__(self, image: np.ndarray) -> np.ndarray:
        """
        Image preprocessing. Transform RGB image to binary image, resizes it and flattens

        :param image: original image
        :return: flatten image
        """

        image = cv2.resize(image, dsize=self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        threshold, bw_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        bw_image_flatten = bw_image.flatten()

        return bw_image_flatten

    def __read_image__(self, image_path) -> np.ndarray:
        """
        Reads image from disk

        :param image_path: path to image
        :return: flatten image
        """

        image = cv2.imread(image_path)

        image = self.__preprocess_image__(image=image)

        return image

    def __get_all_flatten_images__(self) -> List[np.ndarray]:
        """
        Reads all images from list_of_paths

        :return: list of all flatten images
        """

        flatten_images_list = list()

        for curr_path in self.list_of_paths:
            curr_flatten_image = self.__read_image__(image_path=curr_path)

        flatten_images_list.append(curr_flatten_image)

        return flatten_images_list

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Gets image by id in lost_of_paths

        :param idx:
        :return: flatten image
        """

        image_path = self.list_of_paths[idx]

        image = self.__read_image__(image_path=image_path)

        return image

    def __len__(self) -> int:
        """
        Returns length of dataset

        :return: length of dataset
        """

        length = len(self.list_of_paths)

        return length

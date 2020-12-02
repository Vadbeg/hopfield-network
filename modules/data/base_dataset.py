"""Module with base dataset"""

from abc import ABC, abstractmethod


class BaseDataset(ABC):

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass

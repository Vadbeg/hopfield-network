"""Module with Hopfield network"""

from typing import List

import numpy as np
from tqdm import tqdm


class HopfieldNetwork:
    """Hopfiled network implementation"""

    def __init__(self, train_data: List[np.ndarray],
                 asynchronous: bool = False,
                 verbose: bool = True):
        self.train_data = train_data
        self.num_neurons = train_data[0].shape[0]

        self.weights = self.__initialize_weights__(num_neurons=self.num_neurons)

        self.asynchronous = asynchronous
        self.verbose = verbose

    def train(self):
        """Trains models using examples from train_data"""

        rho = self.__get_rho__()

        copied_train_data = np.copy(self.train_data)

        for curr_train_sample in tqdm(copied_train_data,
                                      disable=not self.verbose,
                                      postfix=f'Model training...'):

            train_sample_norm = curr_train_sample - rho

            assert len(train_sample_norm.shape) == 1, \
                f'Flatten your input! Now dim is: {train_sample_norm.shape}'

            self.weights += np.outer(train_sample_norm, train_sample_norm)

        diagonal_values = np.diag(self.weights)  # extracts diagonal values from matrix
        diagonal_weights = np.diag(diagonal_values)  # creates diagonal matrix from diagonal values for weights

        self.weights = self.weights - diagonal_weights
        self.weights = self.weights / len(self.train_data)

    def predict(self, data: List[np.ndarray], num_iter: int = 20,
                threshold: int = 0) -> List[np.ndarray]:
        """
        Predicts data class

        :param data: list of ndarrays with data
        :param num_iter: number of iterations for data passing
        :param threshold: ???
        :return: resulted data
        """

        copied_data = np.copy(data)

        predicted_data = list()

        for curr_copied_sample in tqdm(copied_data,
                                       disable=not self.verbose,
                                       postfix=f'Predicting...'):
            curr_prediction = self.__forward__(initial_data=curr_copied_sample,
                                               threshold=threshold,
                                               num_iter=num_iter)
            predicted_data.append(curr_prediction)

        return predicted_data

    @staticmethod
    def __initialize_weights__(num_neurons: int):
        weights = np.zeros((num_neurons, num_neurons))

        return weights

    def __get_rho__(self):
        train_data_sum = np.sum(self.train_data)

        data_length = len(self.train_data)

        rho = train_data_sum / (data_length * self.num_neurons)

        return rho

    def __forward__(self, initial_data: np.ndarray, threshold: float, num_iter: int = 20) -> np.ndarray:
        copied_initial_data = np.copy(initial_data)

        if self.asynchronous:
            curr_energy = self.__energy__(initial_data=copied_initial_data,
                                          threshold=threshold)

            for iter_idx in range(num_iter):
                for _ in range(100):
                    neuron_idx = np.random.randint(0, self.num_neurons)

                    copied_initial_data[neuron_idx] = np.sign(
                        self.weights[neuron_idx].T.dot(copied_initial_data) - threshold
                    )

                curr_energy_new = self.__energy__(initial_data=copied_initial_data,
                                                  threshold=threshold)

                if curr_energy_new == curr_energy:

                    return copied_initial_data

                curr_energy = curr_energy_new
        else:
            curr_energy = self.__energy__(initial_data=copied_initial_data,
                                          threshold=threshold)

            for iter_idx in range(num_iter):
                copied_initial_data = np.sign(self.weights.dot(copied_initial_data) - threshold)

                curr_energy_new = self.__energy__(initial_data=copied_initial_data,
                                                  threshold=threshold)

                if curr_energy_new == curr_energy:
                    return copied_initial_data

                curr_energy = curr_energy_new

        return copied_initial_data

    def __energy__(self, initial_data: np.ndarray, threshold: float) -> float:
        energy = - 0.5 * initial_data.dot(self.weights).dot(initial_data)
        energy += np.sum(initial_data * threshold)

        return energy


if __name__ == '__main__':
    data = [np.random.random((5, 5)).flatten() for i in range(5)]
    net = HopfieldNetwork(train_data=data)

    net.train()

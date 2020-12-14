"""Module for building plots"""

import glob

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from modules.network.hopfield import HopfieldNetwork
from modules.data.dataset import Dataset
from modules.train import get_trained_model
from config import Config


def get_weights_plot():
    """Creates weights plot"""

    image_paths_train = glob.glob(pathname='images_diff/train/*.*', recursive=True)

    config = Config()
    config.image_size = (10, 10)

    model = get_trained_model(image_paths=image_paths_train,
                              image_size=config.image_size,
                              asynchronous=config.asynchronous)

    plt.title(f'Weight heatmap.')

    print(model.weights)

    sns.heatmap(model.weights)
    plt.show()


def get_image_num_iters_plot():
    """Creates num_iters for images plot"""

    config = Config()
    num_iter_list = [1, 5, 10, 20, 50, 100, 1000]

    image_paths = glob.glob(pathname='images_same/*.*', recursive=True)

    fig, axs = plt.subplots(len(image_paths), len(num_iter_list) + 2)
    [curr_ax.set_axis_off() for curr_ax in axs.ravel()]

    model = get_trained_model(image_paths=image_paths,
                              image_size=config.image_size,
                              asynchronous=config.asynchronous)

    dataset_original = Dataset(list_of_paths=image_paths,
                               image_size=config.image_size,
                               add_noise=False)

    flatten_images = dataset_original.get_all_flatten_images()
    dataset_original.add_noise = True
    flatten_images_noise = dataset_original.get_all_flatten_images()

    axs[0][0].set_title(f'Original image')
    axs[0][1].set_title(f'Noise image')

    for idx_columns, curr_iter_value in enumerate(num_iter_list, start=2):
        axs[0][idx_columns].set_title(curr_iter_value)

    for idx, (curr_flatten_image, curr_flatten_image_noise) in enumerate(
            zip(flatten_images, flatten_images_noise)):

        original_image = np.reshape(curr_flatten_image, newshape=config.image_size)
        noise_image = np.reshape(curr_flatten_image_noise, newshape=config.image_size)

        axs[idx][0].imshow(original_image)
        axs[idx][1].imshow(noise_image)

        for image_idx, curr_num_iter in enumerate(num_iter_list, start=2):
            prediction = model.predict(data=[curr_flatten_image_noise],
                                       num_iter=curr_num_iter,
                                       threshold=config.threshold)[0]

            pred_image = np.reshape(prediction, newshape=config.image_size)

            axs[idx][image_idx].imshow(pred_image)

    fig.suptitle(f'Num of model iterations for different models')
    plt.show()


def get_numbers_example():
    """Builds plot with number calculation example"""

    image_paths_train = glob.glob(pathname='images_diff/train/*.*', recursive=True)
    image_paths_test = glob.glob(pathname='images_diff/test/*.*', recursive=True)

    image_paths_original = ['images_diff/train/6.jpg', 'images_diff/train/8.jpg', 'images_diff/train/0.jpg']

    # image_paths_train_6 = ['images_diff/train2/6/img_442.jpg'] * 3
    # image_paths_train_8 = ['images_diff/train2/8/img_176.jpg'] * 3
    #
    # image_paths_test_6 = glob.glob(pathname='images_diff/test2/6/*.*', recursive=True)
    # image_paths_test_8 = glob.glob(pathname='images_diff/test2/8/*.*', recursive=True)
    #
    # image_paths_train = image_paths_train_6 + image_paths_train_8
    # image_paths_test = image_paths_test_6 + image_paths_test_8

    config = Config()
    config.image_size = (64, 64)
    config.num_iter = 70
    config.threshold = 100

    model = get_trained_model(image_paths=image_paths_train,
                              image_size=config.image_size,
                              asynchronous=config.asynchronous)

    dataset_test = Dataset(list_of_paths=image_paths_test,
                           image_size=config.image_size,
                           add_noise=False)
    flatten_images_test = dataset_test.get_all_flatten_images()

    dataset_original = Dataset(list_of_paths=image_paths_original,
                               image_size=config.image_size,
                               add_noise=False)
    flatten_images_original = dataset_original.get_all_flatten_images()

    predictions = model.predict(data=flatten_images_test,
                                num_iter=config.num_iter,
                                threshold=config.threshold)

    fig, axs = plt.subplots(len(predictions), 3)

    if len(axs.shape) < 2:
        axs = [axs]

    axs[0][0].set_title(f'Original image')
    axs[0][1].set_title(f'Test image')
    axs[0][2].set_title(f'Pred image')

    for idx, (curr_original_image, curr_test_image, curr_pred) in enumerate(zip(
            flatten_images_original, flatten_images_test, predictions)):

        curr_original_image = np.reshape(curr_original_image, newshape=config.image_size)
        curr_test_image = np.reshape(curr_test_image, newshape=config.image_size)
        curr_pred = np.reshape(curr_pred, newshape=config.image_size)

        axs[idx][0].imshow(curr_original_image)
        axs[idx][1].imshow(curr_test_image)
        axs[idx][2].imshow(curr_pred)

    fig.suptitle(f'Result of number predictions')
    plt.show()


def get_energy_plot():
    """Builts plot with energy values"""

    image_paths_train = glob.glob(pathname='images_diff/train/*.*', recursive=True)[:1]
    image_paths_test = glob.glob(pathname='images_diff/test/*.*', recursive=True)[:1]

    config = Config()
    config.image_size = (28, 28)
    config.num_iter = 70
    config.threshold = 100

    model = get_trained_model(image_paths=image_paths_train,
                              image_size=config.image_size,
                              asynchronous=config.asynchronous)

    dataset_test = Dataset(list_of_paths=image_paths_test,
                           image_size=config.image_size,
                           add_noise=False)
    flatten_images_test = dataset_test.get_all_flatten_images()

    predictions = model.predict(data=flatten_images_test,
                                num_iter=config.num_iter,
                                threshold=config.threshold)

    energy_list = model.energy_list

    fig, ax = plt.subplots()
    plt.title(f'Energy values during iterations')

    plt.ylabel(f'Energy')
    plt.xlabel(f'Iter number')

    sns.lineplot(x=range(len(energy_list)), y=energy_list, ax=ax)

    plt.show()


def get_orthogonal_plots():
    config = Config()

    config.asynchronous = False
    config.image_size = (30, 30)

    step = 6

    ones_matrix = np.ones(shape=(step, config.image_size[0]))
    # zeros_matrix = np.zeros(shape=(27, 27))

    orth_images = list()

    for i in range(0, config.image_size[0], step):
        orth_matrix = - np.ones(shape=config.image_size)

        try:
            orth_matrix[i:i + step] = ones_matrix

            orth_images.append(orth_matrix)
        except ValueError as err:
            pass

    orth_images_noise = [Dataset.__change_random_pixels__(curr_orth_image, percent=0.2) for curr_orth_image in orth_images]
    orth_images_flatten = [curr_orth_image.flatten() for curr_orth_image in orth_images]
    orth_images_noise_flatten = [curr_orth_image.flatten() for curr_orth_image in orth_images_noise]

    fig, axs = plt.subplots(len(orth_images), 3)

    if len(axs.shape) < 2:
        axs = [axs]

    model = HopfieldNetwork(train_data=orth_images_flatten,
                            asynchronous=config.asynchronous,
                            projections=config.projections)
    model.train()

    predictions = model.predict(data=orth_images_noise_flatten)

    axs[0][0].set_title(f'Original image')
    axs[0][1].set_title(f'Test image')
    axs[0][2].set_title(f'Pred image')

    for idx, (curr_original_image, curr_test_image, curr_pred) in enumerate(zip(
            orth_images_flatten, orth_images_noise_flatten, predictions)):

        curr_original_image = np.reshape(curr_original_image, newshape=config.image_size)
        curr_test_image = np.reshape(curr_test_image, newshape=config.image_size)
        curr_pred = np.reshape(curr_pred, newshape=config.image_size)

        axs[idx][0].imshow(curr_original_image)
        axs[idx][1].imshow(curr_test_image)
        axs[idx][2].imshow(curr_pred)

    fig.suptitle(f'Result of number predictions')
    plt.show()

    sns.heatmap(model.weights)
    print(model.weights)

    plt.show()

if __name__ == '__main__':
    # get_weights_plot()
    # get_image_num_iters_plot()
    get_numbers_example()
    # get_energy_plot()


    # get_orthogonal_plots()

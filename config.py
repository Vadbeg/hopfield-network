"""Module with config class"""


class Config:
    """Config class"""

    num_iter = 40
    threshold = 50

    image_size = (64, 64)

    asynchronous = True


# количество итерций для одного изображения
num_iter = 40
# спещения порога для функции активации sign
threshold = 50

# размер изображения
image_size = (64, 64)

# при True используется асинхронная модель
# иначе синхронная
asynchronous = True

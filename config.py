"""Module with config class"""


class Config:
    """Config class"""

    num_iter = 100
    threshold = 0

    image_size = (64, 64)

    asynchronous = True
    projections = True


# количество итерций для одного изображения
num_iter = 40

# размер изображения
image_size = (30, 30)

# при True используется асинхронная модель
# иначе синхронная
asynchronous = True

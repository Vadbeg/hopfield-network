# Hopfield network
[Hopfield network](https://en.wikipedia.org/wiki/Hopfield_network) implementations using numpy

It uses numpy for matrix multiplication. 
Here you can't find usage of Deep Learning framework. So this code can be useful for you, if you want to learn 
Hopfield networks from scratch.
Supports only CPU computing.

This code is highly inspired by this
[Hopfield-Network](https://github.com/takyamamoto/Hopfield-Network) project.
takyamamoto, thanks a lot!

## Getting Started

To download project:
```
git clone https://github.com/Vadbeg/hopfield_network.git
```


### Installing
To install all libraries you need, print in `hopfield_network` directory: 

```
pip install -r requirements.txt
```

It will install all essential libraries


### Config

To use project with `start_training.py` or `build_plots.pt` you need to setup config. 
Config is located in `config.py` Config class. Example:

```
class Config:
    """Config class"""

    num_iter = 40
    threshold = 50

    image_size = (64, 64)

    asynchronous = True
```

### Usage 

After libraries installation you are free to use this project. It is great to start from `start_training.py`
script. You can change data folders in this script and start training. Script will show you plots with training result.

Also you can use `build_plots.py` script. It will show you different plots with insights about how Hopfield network works.


## Built With

* [numpy](https://flask.palletsprojects.com/en/1.1.x/) - The math framework used.

## Authors

* **Vadim Titko** aka *Vadbeg* - [GitHub](https://github.com/Vadbeg/PythonHomework/commits?author=Vadbeg) 
| [LinkedIn](https://www.linkedin.com/in/vadtitko/)
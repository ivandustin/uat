from jax.random import split
from .param import param


def create(key, input_dim, width):
    shapes = [(input_dim, width), (1, width), (width,)]
    return tuple(param(key, shape) for key, shape in zip(split(key, 3), shapes))

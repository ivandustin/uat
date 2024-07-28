from jax.random import PRNGKey
from jax.numpy import array
from pytest import fixture
from uat import create


@fixture
def key():
    return PRNGKey(0)


@fixture
def input_dim():
    return 2


@fixture
def width():
    return 2


@fixture
def params(key, input_dim, width):
    return create(key, input_dim, width)


@fixture
def x():
    return array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


@fixture
def y():
    return array([0, 1, 1, 0], dtype=float)

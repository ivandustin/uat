from jax.random import PRNGKey
from jax.numpy import array
from pytest import fixture
from uat import create


@fixture
def key():
    return PRNGKey(0)


@fixture
def neurons():
    return 2


@fixture
def input_dim():
    return 2


@fixture
def output_dim():
    return 1


@fixture
def dimensions(input_dim, neurons, output_dim):
    return (input_dim, neurons, output_dim)


@fixture
def params(key, dimensions):
    return create(key, dimensions)


@fixture
def x():
    return array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


@fixture
def y():
    return array([[0], [1], [1], [0]], dtype=float)

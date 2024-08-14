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
def output_dim():
    return 1


@fixture
def neurons():
    return 2


@fixture
def dims(input_dim, output_dim, neurons):
    return (input_dim, output_dim, neurons)


@fixture
def params(key, dims):
    return create(key, dims)


@fixture
def x():
    return array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


@fixture
def y():
    return array([[0], [1], [1], [0]], dtype=float)

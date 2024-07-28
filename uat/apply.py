from jax.numpy import exp


def apply(params, x):
    a, b, c = params
    return exp((x @ a) + b) @ c

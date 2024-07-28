from optax import sgd, apply_updates
from jax.numpy import allclose
from jax import grad, jit
from uat import apply


def test(params, x, y):
    optimizer = sgd(learning_rate=0.1, momentum=0.9)
    state = optimizer.init(params)
    fit = create(optimizer)
    for _ in range(100):
        state, params = fit(state, params, x, y)
    y_hat = apply(params, x)
    assert allclose(y, y_hat, atol=0.1)


def create(optimizer):
    @jit
    def fit(state, params, x, y):
        grads = grad(loss)(params, x, y)
        updates, state = optimizer.update(grads, state)
        params = apply_updates(params, updates)
        return state, params

    return fit


def loss(params, x, y):
    y_hat = apply(params, x)
    return sum((y - y_hat) ** 2)

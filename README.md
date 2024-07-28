# uat

Universal Approximation Theorem in JAX

## Universal Approximation Theorem

The Universal Approximation Theorem states that a feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, given appropriate activation functions.

### General Formula

The general formula for a neural network with one hidden layer can be expressed as:

$$f(x) = \sum_{i=1}^{N} c_i \sigma(a_i \cdot x + b_i)$$

where:
- $x$ is the input vector.
- $N$ is the number of neurons in the hidden layer.
- $a_i$ and $b_i$ are the weights and biases of the neurons in the hidden layer.
- $c_i$ are the weights of the output layer.
- $\sigma$ is the activation function (e.g., sigmoid, ReLU, exponential).

In this implementation, we use the exponential function as the activation function.

## Usage

### Creating a Model

To create a model, use the `create` function. This function initializes the parameters of the model.

```python
from jax.random import PRNGKey
from uat import create

key = PRNGKey(0)
input_dim = 2
width = 2
params = create(key, input_dim, width)
```

### Applying the Model

To apply the model to an input, use the `apply` function. This function computes the output of the model given the input and the model parameters.

```python
from jax.numpy import array
from uat import apply

x = array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
output = apply(params, x)
```

### Explanation of the `apply` Function

The `apply` function computes the output of the model using the following formula:

```python
def apply(params, x):
    a, b, c = params
    return exp((x @ a) + b) @ c
```

Here's a step-by-step explanation of the formula:

1. **Parameter Unpacking**: The parameters `params` are unpacked into three components: `a`, `b`, and `c`.
   - `a` is a matrix of shape `(input_dim, width)`.
   - `b` is a bias vector of shape `(1, width)`.
   - `c` is a weight vector of shape `(width,)`.

2. **Matrix Multiplication**: The input `x` (of shape `(n_samples, input_dim)`) is multiplied by the matrix `a` using the `@` operator. This results in a matrix of shape `(n_samples, width)`.

3. **Bias Addition**: The bias vector `b` is added to the result of the matrix multiplication. Broadcasting is used to add `b` to each row of the matrix, resulting in a matrix of shape `(n_samples, width)`.

4. **Exponential Activation**: The exponential function `exp` is applied element-wise to the result of the bias addition. This introduces non-linearity into the model.

5. **Output Calculation**: The resulting matrix (of shape `(n_samples, width)`) is then multiplied by the weight vector `c` using the `@` operator. This results in the final output vector of shape `(n_samples,)`.

### Training the Model on XOR

To train the model on the XOR problem, you can use the following code. This code uses stochastic gradient descent (SGD) to optimize the model parameters.

```python
from optax import sgd, apply_updates
from jax.numpy import array, allclose
from jax.random import PRNGKey
from jax import grad, jit
from uat import create, apply

# Initialize parameters
key = PRNGKey(0)
input_dim = 2
width = 2
params = create(key, input_dim, width)

# XOR input and output
x = array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = array([0, 1, 1, 0], dtype=float)

# Define optimizer
optimizer = sgd(learning_rate=0.1, momentum=0.9)
state = optimizer.init(params)

# Define loss function
def loss(params, x, y):
    y_hat = apply(params, x)
    return sum((y - y_hat) ** 2)

# Define training step
@jit
def fit(state, params, x, y):
    grads = grad(loss)(params, x, y)
    updates, state = optimizer.update(grads, state)
    params = apply_updates(params, updates)
    return state, params

# Train the model
for _ in range(100):
    state, params = fit(state, params, x, y)

# Check the output
y_hat = apply(params, x)
assert allclose(y, y_hat, atol=0.1)
```

This code initializes the model parameters, defines the XOR input and output, sets up the optimizer, and trains the model for 100 iterations. Finally, it checks if the model's output is close to the expected XOR output.

### Note on Optimizer Compatibility

Since the `create` function outputs a pytree, you can use any JAX-based optimizer library like `optax` to optimize the model parameters. This allows for flexibility in choosing different optimization algorithms and techniques to train your model effectively.
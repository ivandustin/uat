def test(params, neurons, input_dim, output_dim):
    a, b, c = params
    assert a.shape == (input_dim, neurons)
    assert b.shape == (1, neurons)
    assert c.shape == (neurons, output_dim)

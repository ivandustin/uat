def test(params, input_dim, width):
    a, b, c = params
    assert a.shape == (input_dim, width)
    assert b.shape == (1, width)
    assert c.shape == (width,)

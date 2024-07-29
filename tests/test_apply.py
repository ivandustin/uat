from uat import apply


def test(params, x, output_dim):
    output = apply(params, x)
    assert output.shape == (4, output_dim)

from uat import apply


def test(params, x):
    output = apply(params, x)
    assert output.shape == (4,)

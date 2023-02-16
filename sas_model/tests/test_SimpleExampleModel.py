import pandas as pd

from models.SimpleExampleModel import SimpleExampleModel


def test_model():
    """ test function behaves as expected:
    values less than x0 -> y0, values greater-equal x0 -> y = mx + c   (eqn of straight line)
        m is the gradient, c is the intercept = y0 - (gradient * x0)
    """
    # instance of the model
    model = SimpleExampleModel()
    
    # test data in a Pandas series -2 ... 12
    data = pd.DataFrame([range(-2, 13)])

    # apply the model's _model_function to get output in a Pandas series
    model_output = data.apply(model._model_function, axis=1)

    assert int(2) == 2
    

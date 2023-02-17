import pandas as pd
import pytest

from models.SimpleExampleModel import SimpleExampleModel


# set up some test data as a fixture to pass in to tests
@pytest.fixture
def test_data():

    # create some test data in a data-frame, id 1..4
    test_ids = range(1,5)
    test_age = pd.Series([12,20,70,40])
    test_sex = pd.Series(["m","f","m","f"])
    
    df = pd.DataFrame({'ID':test_ids, 'age':test_age, 'sex': test_sex})

    return df


# get the model results to check
@pytest.fixture
def model_output(test_data):
     # instance of the model
    model = SimpleExampleModel()
    
    # apply the model's _model_function to get output in a Pandas series
    return model._model_function(test_data)


def test_age_grp(model_output):
    """ test age_grp logic behaves as expected"""
    
    # check we get the expexted output for age_grp
    assert all([a == b for a, b in zip(model_output['age_grp'], ['young age', 'young age', 'old age', 'middle age'])])
   

def test_sex(model_output):
    """ test man/woman logic behaves as expected"""

    # check we get the expexted output for sex
    assert all([a == b for a, b in zip(model_output['sex'], ['man', 'woman', 'man', 'woman'])])
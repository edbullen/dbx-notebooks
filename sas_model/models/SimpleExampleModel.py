import mlflow
import pandas as pd
import random


class SimpleExampleModel(mlflow.pyfunc.PythonModel):
    """subclass the MLflow PythonModel class and create a model that takes DF input in the form:
        | ID  | age  | sex |
        --------------------
        |  1  |  12  |  f  |
        |  2  |  35  |  m  |
        |  3  |  75  |  m  |
        |  4  |  15  |  f  |
        --------------------
       and returns DF with three cols: ID, young/middle/old age and woman/man output"""
    
    def __init__(self):
        """intialise the class instance and optionally define extra attributes"""
        

    def _model_function(self, model_input :pd.DataFrame):
        """model code is placed here"""

        rows = len(model_input)

        dummy_ids = range(1,rows+1)
        #dummy_age = pd.Series(random.choice([12,20,35,70,89,40]) for _ in range(rows))
        #dummy_sex = pd.Series(random.choice(["f", "m"]) for _ in range(rows))
        dummy_age_group = pd.Series(["middle_age"]).repeat(rows)
        dummy_sex = pd.Series(["m"]).repeat(rows)

        dummy_data = pd.DataFrame({'ID':dummy_ids, 'age': dummy_age_group, 'sex': dummy_sex })
        
        model_output = dummy_data
        
        return model_output

    def predict(self, context, model_input :pd.DataFrame):
        """create a predict() method for MLflow to use - calls the custom fn to get results"""
        
        return model_input.apply(self._model_function)
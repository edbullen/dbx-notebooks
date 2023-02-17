import mlflow
import pandas as pd
import random


class SimpleExampleModel(mlflow.pyfunc.PythonModel):
    """subclass the MLflow PythonModel class and create a model that takes DF input in the form:
        | ID  | age  | sex |
        --------------------
        |  1  |  12  |  f  |
        |  2  |  35  |  m  |
        | ... |  ... |  ...|
        |  n  |   #  | m/f |
        --------------------
       and returns DF with three cols: ID, young/middle/old age and woman/man output"""
    
    def __init__(self):
        """intialise the class instance and optionally define extra attributes"""
        

    def _model_function(self, model_input :pd.DataFrame):
        """model code is placed here"""

        def age_grp_if(x): 
            """ inner function - age-class logic """
            if (x <= 35) :
                return 'young age'
            elif (x <= 60):
                return 'middle age'
            else:
                return 'old age'
            
        def woman_man_if(x): 
            """ inner function - man-woman-class logic """
            if (x == 'f') :
                return 'woman'
            elif (x == 'm'):
                return 'man'
            else:
                return 'other'
    
        # accumulate output data in lists using Pandas apply-function.  Could re-write to pandas.loc operation for speedup.        
        age_grp = model_input['age'].apply(age_grp_if)
        sex = model_input['sex'].apply(woman_man_if)

        return pd.DataFrame({'ID':model_input['ID'], 'age_grp':age_grp, 'sex':sex})    


    def predict(self, context, model_input :pd.DataFrame):
        """create a predict() method for MLflow to use - calls the custom fn to get results"""
        
        return self._model_function(model_input)
# Migrate SAS Models to Python with MLflow 

Example demonstating core [SAS model logic re-written in Python](https://towardsdatascience.com/how-to-use-if-then-else-in-python-the-way-you-work-in-sas-3ccbda30ce8b) using procedural SAS `IF`, `ELSE`, `ELSE IF` logic copied to Python `if`, `else`, `elif` logic.  

MLflow can be used to package up the code and serve it as a model with a predict method in a model served by MLflow.  This provides the following functionality:  
+ surface the model-code logic with a predict interface (pass in a Pandas Dataframe, return prediction result-set)  
+ track different versions of the model in MLflow; access different versions in the MLflow name-space  
+ Get predictions from the model to either using *Batch Scoring in SQL* with a UDF function or via *Rest API* call with MLflow model-serving  
  - For batch scoring, the MLflow model is linked to a UDF and surfaced as a SQL function applied to a data-table of inputs   
  - For rest API calls, the MLflow model is served out of the MLflow REST API model serving service, and input data provided as a HTTP POST request with an authentication token in the REST API header.  


1. Create the core model functionality in a Python Class.  This needs to sub-class the MLflow [`pyfunc.PythonModel`] class(https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and re-implement the `predict` method (see example below)  

2. Create a custom method that implements the SAS model functionality.  This should be written to take a Pandas Dataframe input and return a DataFrame of the prediction results.  

   
```python
class SASModelClass(mlflow.pyfunc.PythonModel):
    """subclass the MLflow PythonModel class"""
    
    def __init__(self):
        """intialise the class instance and optionally define extra attributes"""
        
        <set attrs>

    def _model_function(self, model_input :pd.DataFrame):
        """model code is placed here"""
        
        <custom model code here>
        
        return model_output

    def predict(self, context, model_input :pd.DataFrame):
        """create a predict() method for MLflow to use - calls the custom fn to get results"""
        
        return model_input.apply(self._custom_function)
```

3. Create an empty `__init__.py` file in the Class folder
4. Use unit-tests to confirm the model_function works as planned then deply in MLflow and test the predition method.  


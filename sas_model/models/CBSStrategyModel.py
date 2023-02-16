class CSBStrategyModel(mlflow.pyfunc.PythonModel):
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
# Databricks notebook source
# DBTITLE 1,Deploy Custom Model to MLFlow
import mlflow

# COMMAND ----------

# DBTITLE 1,Import Model in Python Class from Repo
from models.SimpleExampleModel import SimpleExampleModel

# COMMAND ----------

# DBTITLE 1,Instantiate the Model
model = SimpleExampleModel()

# COMMAND ----------

# DBTITLE 1,Load Python Requirements from Repo 
requirements_file = '/Workspace/Repos/ed.bullen@databricks.com/dbx-notebooks/requirements.txt'
with open(requirements_file) as file:
    requirements_list = [line.rstrip() for line in file]


# COMMAND ----------

# DBTITLE 1,Log the model as an MLflow Experiment
model_req_list = ["pandas==1.5.2"]

mlflow.set_experiment("/Users/ed.bullen@databricks.com/simple_example_model")
response = mlflow.pyfunc.log_model("simple_example_model", python_model=model, pip_requirements=requirements_list)

# COMMAND ----------

# DBTITLE 1,Set Tags, Metrics (parameters) and Description to store in MLflow 
# Set any tags we want in the Run to help identify it
mlflow.set_tag("project", "simple_example_model")
mlflow.set_tag("mlflow.note.content", "Example of migrating a simple SAS model to MLflow")

# Set some metrics that provide information about the run - i.e. the parameters associated with the model-instance
#mlflow.log_metric('x', 1)

mlflow.end_run() 

# COMMAND ----------

# DBTITLE 1,Unique reference for this experiment run

print(f"Artifact Path: \t {response.artifact_path}")
print(f"Run ID: \t {response.run_id}")
print(f"Model URI: \t {response.model_uri}")

# COMMAND ----------

# DBTITLE 1,Register the Model in the MLflow Model Registry
# register the model
registry_response = mlflow.register_model(response.model_uri, "simple_example_model")

# update some meta-data details about the model such as the description
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.update_model_version(
    version=registry_response.version,
    name = registry_response.name,
    description="This is a simple example of a SAS model re-written in Python and served out of MLflow"
)

# COMMAND ----------

print(f"Model Version Number: \t {registry_response.version}")

# COMMAND ----------

# DBTITLE 1,Transition Model to Staging
client.transition_model_version_stage("simple_example_model", registry_response.version, stage = "Staging", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC **Staging Model name-space**
# MAGIC ```
# MAGIC models:/custom_model/Staging
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Transition Version 1 to Production
client.transition_model_version_stage("simple_example_model", 1, stage = "Production", archive_existing_versions=True)

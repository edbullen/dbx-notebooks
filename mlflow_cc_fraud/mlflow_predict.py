# Databricks notebook source
# MAGIC %md
# MAGIC # Predictions using ML Models managed by MLflow
# MAGIC Example scenarios demonstrated
# MAGIC 1. Get predictions using the **MLflow Rest API** 
# MAGIC 2. Apply a **Databricks MLflow model** using **SQL** against data queried from **BigQuery** - "batch scoring"
# MAGIC 3. Save results to **Delta Table**  - batch results available for downstream analytics
# MAGIC 
# MAGIC *Many other options and approaches are possible* 
# MAGIC - EG streaming data as it arrives against the model and streaming the predictions out to a results table.
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1gdajj0xkl5Y2-fKaIyVOfthbKYZwyf7O" alt="drawing" width="600"/>

# COMMAND ----------

# DBTITLE 1,Demo Env Preparation
# set some vars to generate names for where we store our experiments and feature data
import re
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

db_name = current_user_no_at
# naming convention is <initials>_catalog (for Unity Catalog)
catalog_name = current_user_no_at.split("_")[0][0] + current_user_no_at.split("_")[1][0] + '_catalog'

spark.conf.set("var.db_name",db_name)
spark.conf.set("var.catalog_name",catalog_name)
spark.conf.set("var.current_user",current_user)

print(f"catalog_name set to {catalog_name}")
print(f"db_name set to {db_name}")
print(f"current_user is {current_user}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. MLflow Rest API Predictions
# MAGIC 
# MAGIC Format the dataset to be scored in a Pandas Dataframe:
# MAGIC 
# MAGIC ```
# MAGIC payload_df = pd.DataFrame([[1.226073, -1.640026, ... , 0.012658],
# MAGIC                           ],
# MAGIC                           columns=["pca[0]", "pca[1]", ... "pca[27]"])
# MAGIC ```                          
# MAGIC 
# MAGIC Create a `score_model()` function and call it providing the MLflow URL, access-token and data to process:
# MAGIC ```
# MAGIC def score_model(api_url, token, dataset):
# MAGIC     headers = {'Authorization': f'Bearer {token}'}
# MAGIC 
# MAGIC     data_core = dataset.to_dict(orient='records')
# MAGIC     data_json = {"dataframe_records": data_core}
# MAGIC     
# MAGIC     response = requests.request(method='POST', headers=headers, url=api_url, json=data_json)
# MAGIC     if response.status_code != 200:
# MAGIC         raise Exception(f'Request failed with status {response.status_code}, {response.text}')
# MAGIC     return response.json()
# MAGIC 
# MAGIC ```
# MAGIC 
# MAGIC Sample code is provided in this repo - see `./mlflow_cc_fraud/mlflow_rest_call.py`

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. SQL Batch Score with a PySpark UDF
# MAGIC "UDF" is a User Defined Function.

# COMMAND ----------

# DBTITLE 1,Load the model and create a UDF function on it - 5 mins
import mlflow

# create a user-defined function that maps to MLflow model
fraud_detect_udf = mlflow.pyfunc.spark_udf(spark, "models:/cc_fraud/Production", env_manager="virtualenv")
#Save the inference as a sql function that can be accessed in Spark / SQL
spark.udf.register("cc_fraud_model", fraud_detect_udf)

# COMMAND ----------

# DBTITLE 1,Query the CC Fraud Silver Table and run Predictions - Sample Subset
# MAGIC %sql 
# MAGIC SELECT `id`, `time`, `amountRange`, 
# MAGIC         cc_fraud_model(
# MAGIC               `pca0`, `pca1`, `pca2`, `pca3`, `pca4`, `pca5`,
# MAGIC               `pca6`, `pca7`, `pca8`, `pca9`, `pca10`, `pca11`, `pca12`,
# MAGIC               `pca13`, `pca14`, `pca15`, `pca16`, `pca17`, `pca18`,
# MAGIC               `pca19`, `pca20`, `pca21`, `pca22`, `pca23`, `pca24`,
# MAGIC               `pca25`, `pca26`, `pca27`) 
# MAGIC        as prediction
# MAGIC --FROM eb_catalog.ed_bullen.cc_fraud_silver
# MAGIC FROM ${var.catalog_name}.${var.db_name}.cc_fraud_silver
# MAGIC WHERE id > 33 AND id < 100
# MAGIC LIMIT 10;

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Batch Write Predictions to Gold Table

# COMMAND ----------

# DBTITLE 1,Store Predictions in Gold Table business analysis
# MAGIC %sql 
# MAGIC INSERT INTO ${var.catalog_name}.${var.db_name}.cc_fraud_gold
# MAGIC SELECT `id`, `time`, `amountRange`, 
# MAGIC         cc_fraud_model(
# MAGIC               `pca0`, `pca1`, `pca2`, `pca3`, `pca4`, `pca5`,
# MAGIC               `pca6`, `pca7`, `pca8`, `pca9`, `pca10`, `pca11`, `pca12`,
# MAGIC               `pca13`, `pca14`, `pca15`, `pca16`, `pca17`, `pca18`,
# MAGIC               `pca19`, `pca20`, `pca21`, `pca22`, `pca23`, `pca24`,
# MAGIC               `pca25`, `pca26`, `pca27`) 
# MAGIC        as prediction
# MAGIC FROM ${var.catalog_name}.${var.db_name}.cc_fraud_silver

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM ${var.catalog_name}.${var.db_name}.cc_fraud_gold
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Find Transactions with Fraud
# MAGIC %sql 
# MAGIC SELECT * FROM eb_catalog.ed_bullen.cc_fraud_gold
# MAGIC WHERE prediction = 1;

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM ${var.catalog_name}.${var.db_name}.cc_fraud_gold
# MAGIC WHERE prediction = 0;

# COMMAND ----------



# COMMAND ----------



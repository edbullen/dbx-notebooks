# Databricks notebook source
# MAGIC %md
# MAGIC # Predictions using ML Models managed by MLflow
# MAGIC Example scenarios demonstrated
# MAGIC 1. Get predictions using the MLflow Rest API (demonstrated in the previous "build" notebook)
# MAGIC 2. Python code generates predictions against a data-frame held in memory 
# MAGIC 3. Apply a model against a database table using SQL - "batch scoring"
# MAGIC 4. Save results to GCP bigtable
# MAGIC 5. Overview of using Vertex AI endpoint and Python API example
# MAGIC 
# MAGIC Other options not demonstrated:
# MAGIC + Streaming data in from Kafka / Pub-Sub, getting predictions, writing out to a Delta database table or a Kafka / Pub-Sub queue
# MAGIC + Direct query of BigQuery for batch-scoring in Databricks
# MAGIC + Writing results out in CSV form to a cloud storage bucket
# MAGIC + Deploying MLflow model artefacts to separate containers  
# MAGIC .  
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1gdajj0xkl5Y2-fKaIyVOfthbKYZwyf7O" alt="drawing" width="600"/>

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Python data-frame predictions

# COMMAND ----------

# DBTITLE 1,Create data-frame X_test_small of test data to score
import pandas as pd
from utils.sample_data import cc_fraud_dict  # helper function to generate some data for getting predictions for

# get a Python dictionary of data
data_dict = cc_fraud_dict()
# convert it to a Pandas data-frame
X_test_small = pd.DataFrame(data_dict)

# COMMAND ----------

display(X_test_small)

# COMMAND ----------

#logged_model = 'runs:/c99bd96b0f4e4bfcbbcf958cd9bc4477/cc_fraud'
logged_model = 'models:/cc_fraud/Production'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(X_test_small)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ## 3. SQL Batch Score with Python UDF

# COMMAND ----------

# DBTITLE 1,Load the model and create a UDF function on it
import mlflow

# create a user-defined function that maps to MLflow model
fraud_detect_udf = mlflow.pyfunc.spark_udf(spark, "models:/cc_fraud/Production")
#Save the inference as a sql function that can be accessed in SPark / SQL
spark.udf.register("cc_fraud_model", fraud_detect_udf)

# COMMAND ----------

# DBTITLE 1,Test data to demonstrate generating predictions - cc_fraud_test
# MAGIC %sql 
# MAGIC SELECT * FROM cc_fraud_test LIMIT 5;

# COMMAND ----------

# DBTITLE 1,Validations data to check the generated predictions - cc_fraud_valid
# MAGIC %sql
# MAGIC SELECT * FROM cc_fraud_valid LIMIT 5;

# COMMAND ----------

# DBTITLE 1,Use the cc_fraud function to extract predictions from data stored in a SQL table
# MAGIC %sql 
# MAGIC SELECT `id`, `time`, `amountRange`, cc_fraud_model(
# MAGIC        `pca[0]`, `pca[1]`, `pca[2]`, `pca[3]`, `pca[4]`, `pca[5]`,
# MAGIC        `pca[6]`, `pca[7]`, `pca[8]`, `pca[9]`, `pca[10]`, `pca[11]`, `pca[12]`,
# MAGIC        `pca[13]`, `pca[14]`, `pca[15]`, `pca[16]`, `pca[17]`, `pca[18]`,
# MAGIC        `pca[19]`, `pca[20]`, `pca[21]`, `pca[22]`, `pca[23]`, `pca[24]`,
# MAGIC        `pca[25]`, `pca[26]`, `pca[27]`) as prediction
# MAGIC FROM cc_fraud_test
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Create a Temp View of Predictions
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW cc_fraud_predictions
# MAGIC AS SELECT `id`, `time`, `amountRange`, cc_fraud_model(
# MAGIC        `pca[0]`, `pca[1]`, `pca[2]`, `pca[3]`, `pca[4]`, `pca[5]`,
# MAGIC        `pca[6]`, `pca[7]`, `pca[8]`, `pca[9]`, `pca[10]`, `pca[11]`, `pca[12]`,
# MAGIC        `pca[13]`, `pca[14]`, `pca[15]`, `pca[16]`, `pca[17]`, `pca[18]`,
# MAGIC        `pca[19]`, `pca[20]`, `pca[21]`, `pca[22]`, `pca[23]`, `pca[24]`,
# MAGIC        `pca[25]`, `pca[26]`, `pca[27]`) as prediction
# MAGIC FROM cc_fraud_test;

# COMMAND ----------

# DBTITLE 1,Query the Validation table, with real fraud labels
# MAGIC %sql
# MAGIC SELECT `id`, `time`, `amountRange`, `label` 
# MAGIC FROM cc_fraud_valid
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Join the Predictions with the Validation data and check the accuracy
# MAGIC %sql
# MAGIC SELECT p.id, p.prediction, p.time, p.amountRange, v.label
# MAGIC FROM cc_fraud_predictions p
# MAGIC JOIN cc_fraud_valid v ON p.id = v.id
# MAGIC WHERE prediction != v.label;

# COMMAND ----------

# DBTITLE 1,Example using SQL to get the predictions we are interested in 
# MAGIC %sql
# MAGIC SELECT `id`, `time`, `amountRange` FROM  cc_fraud_predictions 
# MAGIC WHERE amountRange > 6
# MAGIC AND prediction = 1;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Google BigTable Results Persist
# MAGIC 
# MAGIC + Dependancy: need Python library `google-cloud-bigtable` installed (add it to the cluster configuration)  
# MAGIC    
# MAGIC 1. Create an HBASE / BigTable table: instance = `fieldeng-bigtable-demo-central`, region `us-central-1a`, table = `cc_fraud_bigtable`     
# MAGIC   a) column groups `features`, `predict`  - each column group has id `time` column   
# MAGIC 2. Get the data into a list of lists to insert into the table.  
# MAGIC 3. Use Google HBASE python API to write the data.
# MAGIC 
# MAGIC 
# MAGIC Need to build a data structure in Python (collected in memory in the Spark Driver) as Pyspark interface for Google BigTable not available - https://issuetracker.google.com/issues/227495195
# MAGIC 
# MAGIC Batch Writes in Python using a MutateRows API request: 
# MAGIC https://cloud.google.com/bigtable/docs/writing-data#batch

# COMMAND ----------

# MAGIC %md
# MAGIC Set up a table in BigTable using the Cloud Admin Console
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=10D-GOMl_vgfQ3-fWrFvHJC7mgd1j3GHk" alt="drawing" width="1200"/>

# COMMAND ----------

# DBTITLE 1,Google Cloud API Imports
# BigTable API imports
from google.cloud import bigtable
from google.cloud.bigtable import column_family
from google.cloud.bigtable import row_filters


# COMMAND ----------

# DBTITLE 1,BigTable Client Connect
# Column family and cols are assumed to be already set-up in the table

# initialise client object, instance and table references
client = bigtable.Client(project="fe-dev-sandbox", admin=True)
instance = client.instance("fieldeng-bigtable-demo-central")
table = instance.table("cc_fraud_bigtable")

# COMMAND ----------

# DBTITLE 1,Get a PySpark Dataframe to Write Out
#from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

predictions_df = spark.sql("select id, time, amountRange, prediction FROM cc_fraud_predictions ")
predictions_df = predictions_df.withColumn("prediction", predictions_df["prediction"].cast(IntegerType())) 

# COMMAND ----------

predictions_df.show(5)

# COMMAND ----------

# DBTITLE 1,Write to BigTable in Iterative Loop
# Maximum number of mutations is 100000 per batch in the loop

import datetime
timestamp = datetime.datetime.utcnow()

feature_cols = ["time", "amountRange"]

column_family_1 = "predict"
#column_family_2 = "features"

predictions_pd_df = predictions_df.toPandas()
#predictions_pd_df.memory_usage(index=True, deep=True).sum()/1024/1024
    
row_ids = list(predictions_pd_df['id'])

#list  - predictions
predict_vals = predictions_pd_df["prediction"].astype(int)


feature_cols=list(predictions_df.columns)[1:][:-1]
# pad the feature-cols with zeros
#feature_cols = []

#for i, c in enumerate(feature_cols_pandas):
#    #prefix = "pca"
#    #num = str(i).zfill(3)
#    feature_cols.append(f"{prefix}_{num}")

print(feature_cols)   
    

#list of lists - features
feature_vals = predictions_pd_df[feature_cols].to_numpy().tolist()



# break this up into sets of batch-size
batch_size = 10000
counter = 0

hbase_row_ids = []
# Create some HBASE row-ids from the list of row ids
for i, r in enumerate(row_ids):
    # when counter = batch-size, write out the  hbase_row_ids, and re-initialise a new one (and zero the counter)

    # hbase row identifier
    hbase_row_ids.append( table.direct_row(f"{r}") )
    
    # for each hbase row id, write a prediction and a timestamp
    hbase_row_ids[counter].set_cell(column_family_1, "prediction", f"{predict_vals[i]}", timestamp)              
                            
    # for each hbase row id, write a set of feature-value pairs and a timestamp
    for j,f in enumerate(feature_cols):
        # issue with storing float values - store a strings
        hbase_row_ids[counter].set_cell(column_family_1, f, f"{feature_vals[i][j]}", timestamp)  
       
    # use BigTable mutate rows to write a batch out    
    if counter >= batch_size:
        response = table.mutate_rows(hbase_row_ids)
        hbase_row_ids = []
        print(f"Wrote {counter} rows")
        counter = 0
        
        for n,status in enumerate(response):
            if status.code != 0:
                print("Error writing row: {}".format(status.message))
        
    else:
        counter = counter + 1 

# do a final write for the remainder rows
response = table.mutate_rows(hbase_row_ids)
print(f"Wrote {counter} rows")
        
for n,status in enumerate(response):
    if status.code != 0:
        print("Error writing row: {}".format(status.message))
        
print(f"total records written {i}")    


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Example using the cbt tool to query data from HBASE**
# MAGIC 
# MAGIC Count
# MAGIC ```
# MAGIC cbt -instance=fieldeng-bigtable-demo-central count cc_fraud_bigtable
# MAGIC ```
# MAGIC 
# MAGIC Get a row id=3   *(id = 7120 is a Fraud example)*
# MAGIC ```
# MAGIC cbt -instance=fieldeng-bigtable-demo-central lookup cc_fraud_bigtable 3
# MAGIC ```
# MAGIC 
# MAGIC Get a specific row-id and column
# MAGIC 
# MAGIC ```
# MAGIC cbt -instance=fieldeng-bigtable-demo-central lookup cc_fraud_bigtable 9996 columns=amountRange
# MAGIC ```
# MAGIC 
# MAGIC delete all data
# MAGIC ```
# MAGIC cbt -instance=fieldeng-bigtable-demo-central deleteallrows cc_fraud_bigtable
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Vertex AI 
# MAGIC There is a 1.5 MB limit on the API
# MAGIC 
# MAGIC This part needs cluster version 10.4 - got a protobuf error for 11.4

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vertex in Databricks via the plugin from the notebook - batch inference
# MAGIC 
# MAGIC Pass in a Pandas data-frame with features to produce a prediction ("score") for.

# COMMAND ----------

X_test_small.iloc[:,1:]

# COMMAND ----------

import mlflow
from mlflow import deployments
from mlflow.deployments import get_deploy_client

# create a Vertex AI client
vtx_client = mlflow.deployments.get_deploy_client("google_cloud")

# Use the .predict() method from the same plugin
deployment_name = "mlflow_vertex_ccfraud"
# strip off the id column ans pass in the features (column 1..n)
predictions = vtx_client.predict(deployment_name,X_test_small.iloc[:,1:])

# COMMAND ----------

predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vertex AI Python interface
# MAGIC 
# MAGIC 
# MAGIC 1. Follow the setup instructions in the [Python Client for Cloud AI Platform quick start](https://googleapis.dev/python/aiplatform/latest/index.html) 
# MAGIC 
# MAGIC 
# MAGIC 2. Copy and paste the sample code on Github into a new Python file: [https://github.com/googleapis/python-aiplatform/blob/main/samples/snippets/prediction_service/predict_custom_trained_model_sample.py](https://github.com/googleapis/python-aiplatform/blob/main/samples/snippets/prediction_service/predict_custom_trained_model_sample.py)
# MAGIC 
# MAGIC 
# MAGIC 3. Execute your request in Python.
# MAGIC ```
# MAGIC predict_custom_trained_model_sample(
# MAGIC     project="697856052963",
# MAGIC     endpoint_id="6798082482446008320",
# MAGIC     location="us-central1",
# MAGIC     instance_dict={ "instance_key_1": "value", ...}
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vertex AI Rest API Instructions
# MAGIC 
# MAGIC Copied from the Vertex AI Model Registry deployed-version examples.
# MAGIC 
# MAGIC 1. Make sure you have the Google Cloud SDK  installed.
# MAGIC 2. Run the following command to authenticate with your Google account: 
# MAGIC ```
# MAGIC gcloud auth application-default login
# MAGIC ```
# MAGIC 3. Create a JSON object to hold your data.
# MAGIC ```
# MAGIC {
# MAGIC   "instances": [
# MAGIC     { "instance_key_1": "value", ... }, ...
# MAGIC   ],
# MAGIC   "parameters": { "parameter_key_1": "value", ... }, ...
# MAGIC }
# MAGIC ```
# MAGIC 4. Create environment variables to hold your endpoint and project IDs, as well as your JSON object.
# MAGIC ```
# MAGIC ENDPOINT_ID="6798082482446008320"
# MAGIC PROJECT_ID="fe-dev-sandbox"
# MAGIC INPUT_DATA_FILE="INPUT-JSON"
# MAGIC ```
# MAGIC 5. Execute the request.
# MAGIC ```
# MAGIC curl \
# MAGIC -X POST \
# MAGIC -H "Authorization: Bearer $(gcloud auth print-access-token)" \
# MAGIC -H "Content-Type: application/json" \
# MAGIC https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict \
# MAGIC -d "@${INPUT_DATA_FILE}"
# MAGIC ```

# Databricks notebook source
# MAGIC %md
# MAGIC ## Test Batch Data

# COMMAND ----------

# DBTITLE 1,Create data-frame X_test_small of test data to score
import pandas as pd
from utils.sample_data import cc_fraud_dict

data_dict = cc_fraud_dict()
X_test_small = pd.DataFrame(data_dict)

# COMMAND ----------

display(X_test_small)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## SQL Batch Score with Python UDF

# COMMAND ----------

# DBTITLE 1,Load the model and run inferences in batch (python/SQL/Scala)
import mlflow

# create a user-defined function that maps to MLflow model
fraud_detect_udf = mlflow.pyfunc.spark_udf(spark, "models:/cc_fraud/Production")
#Save the inference as a sql function that can be accessed in SPark / SQL
spark.udf.register("cc_fraud_model", fraud_detect_udf)

# COMMAND ----------

# DBTITLE 1,Create a SQL view of test data to run predictions for
# create a Spark dataframe from the Pandas DF
spark_df=spark.createDataFrame(X_test_small) 

# create a Temp View on the the Spark DF
spark_df.createOrReplaceTempView("X_small_sql")

# COMMAND ----------

# MAGIC %sql
# MAGIC select *  from X_small_sql

# COMMAND ----------

# DBTITLE 1,Use the cc_fraud function to extract predictions from data stored in a SQL table
# MAGIC %sql 
# MAGIC select id, cc_fraud_model(
# MAGIC        `pca[0]`, `pca[1]`, `pca[2]`, `pca[3]`, `pca[4]`, `pca[5]`,
# MAGIC        `pca[6]`, `pca[7]`, `pca[8]`, `pca[9]`, `pca[10]`, `pca[11]`, `pca[12]`,
# MAGIC        `pca[13]`, `pca[14]`, `pca[15]`, `pca[16]`, `pca[17]`, `pca[18]`,
# MAGIC        `pca[19]`, `pca[20]`, `pca[21]`, `pca[22]`, `pca[23]`, `pca[24]`,
# MAGIC        `pca[25]`, `pca[26]`, `pca[27]`) as prediction
# MAGIC from X_small_sql

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Vertex AI 
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## BigTable write from SQL Batch Score

# COMMAND ----------

# DBTITLE 1,Delta Table based on Parquet raw data
# MAGIC %sql
# MAGIC DROP TABLE default.cc_features;
# MAGIC 
# MAGIC CREATE TABLE default.cc_features
# MAGIC USING parquet
# MAGIC OPTIONS (path "/databricks-datasets//credit-card-fraud/data/");

# COMMAND ----------

# MAGIC %sql
# MAGIC describe cc_features;
# MAGIC 
# MAGIC select * from cc_features limit 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM cc_features;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Source Data and Prepare
# MAGIC + Pretend `time` is the ID column  - query the range of values we want - SQL in pyspark
# MAGIC + All the features are in `pcaVector` - Use Python to extract the vector into columns in a PySpark dataframe, then make this into a temp table we can query in SQL

# COMMAND ----------

# DBTITLE 1,Pyspark SQL query into Pyspark DF
cc_pyspark = sqlContext.sql('select * from cc_features \
                            where time > 4000 \
                            and time < 40000 \
                            ')

# COMMAND ----------

cc_pyspark.count()

# COMMAND ----------

# DBTITLE 1,Expand out the Columns in Pyspark
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

cc_pyspark_flat = (cc_pyspark.withColumn("pca", vector_to_array("pcaVector"))).select(["time"] + [col("pca")[i] for i in range(28)])

# COMMAND ----------

display(cc_pyspark_flat.head(5))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Batch Score Results
# MAGIC 
# MAGIC + Create a UDF (function) to apply the model stored in MLflow.   
# MAGIC + Map a SQL view over the spark dataframe and run the UDF against it  
# MAGIC + This example just uses `time` as an ID column to map the predictions back to the original data-set.  

# COMMAND ----------

import mlflow
import warnings
warnings.filterwarnings('ignore')

# create a user-defined function that maps to MLflow model
fraud_detect_udf = mlflow.pyfunc.spark_udf(spark, "models:/cc_fraud/Production")
# register the model as a sql function that can be accessed in SPark / SQL
spark.udf.register("cc_fraud_model", fraud_detect_udf)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Create a temporary SQL table view on the Pyspark DataFrame
cc_pyspark_flat.createOrReplaceTempView("cc_flat_temp")

# COMMAND ----------

# DBTITLE 1,Get Predictions using Model via UDF
# MAGIC %sql 
# MAGIC SELECT time, cc_fraud_model(
# MAGIC        `pca[0]`, `pca[1]`, `pca[2]`, `pca[3]`, `pca[4]`, `pca[5]`,
# MAGIC        `pca[6]`, `pca[7]`, `pca[8]`, `pca[9]`, `pca[10]`, `pca[11]`, `pca[12]`,
# MAGIC        `pca[13]`, `pca[14]`, `pca[15]`, `pca[16]`, `pca[17]`, `pca[18]`,
# MAGIC        `pca[19]`, `pca[20]`, `pca[21]`, `pca[22]`, `pca[23]`, `pca[24]`,
# MAGIC        `pca[25]`, `pca[26]`, `pca[27]`) as prediction
# MAGIC FROM cc_flat_temp

# COMMAND ----------

# DBTITLE 1,Alternatively just get the items that are predicted to be fraud
# MAGIC %sql
# MAGIC SELECT time, prediction
# MAGIC FROM
# MAGIC   (SELECT time, cc_fraud_model(
# MAGIC        `pca[0]`, `pca[1]`, `pca[2]`, `pca[3]`, `pca[4]`, `pca[5]`,
# MAGIC        `pca[6]`, `pca[7]`, `pca[8]`, `pca[9]`, `pca[10]`, `pca[11]`, `pca[12]`,
# MAGIC        `pca[13]`, `pca[14]`, `pca[15]`, `pca[16]`, `pca[17]`, `pca[18]`,
# MAGIC        `pca[19]`, `pca[20]`, `pca[21]`, `pca[22]`, `pca[23]`, `pca[24]`,
# MAGIC        `pca[25]`, `pca[26]`, `pca[27]`) as prediction
# MAGIC   FROM cc_flat_temp) t
# MAGIC WHERE t.prediction = 1;

# COMMAND ----------

# DBTITLE 1,Get the Predictions and Join them back to the Original Dataset
results_df = spark.sql("""
SELECT time, cc_fraud_model(
       `pca[0]`, `pca[1]`, `pca[2]`, `pca[3]`, `pca[4]`, `pca[5]`,
       `pca[6]`, `pca[7]`, `pca[8]`, `pca[9]`, `pca[10]`, `pca[11]`, `pca[12]`,
       `pca[13]`, `pca[14]`, `pca[15]`, `pca[16]`, `pca[17]`, `pca[18]`,
       `pca[19]`, `pca[20]`, `pca[21]`, `pca[22]`, `pca[23]`, `pca[24]`,
       `pca[25]`, `pca[26]`, `pca[27]`) as prediction
FROM cc_flat_temp
""")

cc_pyspark_flat = cc_pyspark_flat.join(results_df, how="inner", on="time")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cc_flat_temp LIMIT 5;

# COMMAND ----------

display(cc_pyspark_flat.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to BigTable
# MAGIC 
# MAGIC + Dependancy: need Python library `google-cloud-bigtable` installed (add it to the cluster configuration)  
# MAGIC    
# MAGIC    
# MAGIC    
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

# BigTable API imports
from google.cloud import bigtable
from google.cloud.bigtable import column_family
from google.cloud.bigtable import row_filters


# COMMAND ----------

# Column family and cols are assumed to be already set-up in the table

# initialise client object, instance and table references
client = bigtable.Client(project="fe-dev-sandbox", admin=True)
instance = client.instance("fieldeng-bigtable-demo-central")
table = instance.table("cc_fraud_bigtable")


# COMMAND ----------

# DBTITLE 1,Batch-Write to HBASE- this takes 4 minutes if we write the features out, 20 seconds if we write the predictions
#Maximum number of mutations is 100000

import datetime
timestamp = datetime.datetime.utcnow()
column_family_1 = "features"
column_family_2 = "predict"

cc_pandas_flat = cc_pyspark_flat.toPandas()
#cc_pandas_flat.memory_usage(index=True, deep=True).sum()/1024/1024

feature_cols_pandas=list(cc_pandas_flat.columns)[1:][:-1]
# pad the feature-cols with zeros
feature_cols = []
for i, c in enumerate(feature_cols_pandas):
    prefix = "pca"
    num = str(i).zfill(3)
    feature_cols.append(f"{prefix}_{num}")

print(feature_cols)   
    
row_ids = list(cc_pandas_flat['time'])
#list of lists - features
feature_vals = cc_pandas_flat[feature_cols_pandas].to_numpy().tolist()
#list  - predictions
predict_vals = cc_pandas_flat["prediction"].astype(int)

# break this up into sets of batch-size
batch_size = 1000
counter = 0

hbase_row_ids = []
# Create some HBASE row-ids from the list of row ids
for i, r in enumerate(row_ids):
    # when counter = batch-size, write out the  hbase_row_ids, and re-initialise a new one (and zero the counter)

    # hbase row identifier
    hbase_row_ids.append( table.direct_row(f"{r}") )
    
    # for each hbase row id, write a prediction and a timestamp
    hbase_row_ids[counter].set_cell(column_family_2, "prediction", f"{predict_vals[i]}", timestamp)              
                            
    # for each hbase row id, write a set of feature-value pairs and a timestamp
    for j,f in enumerate(feature_cols):
        # issue with storing float values??? store a strings
        hbase_row_ids[counter].set_cell(column_family_1, f, f"{feature_vals[i][j]}", timestamp)  
        
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

cc_pandas_flat[cc_pandas_flat['time']==34687]

# COMMAND ----------

for i, status in enumerate(response):
        if status.code != 0:
            print("Error writing row: {}".format(status.message))


# COMMAND ----------

# MAGIC %md
# MAGIC Example using the cbt tool to query data from HBASE
# MAGIC ```
# MAGIC cbt -instance=fieldeng-bigtable-demo-central read cc_fraud_bigtable start=38648 end=38649
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC Get a specific row-id (time=34687)
# MAGIC ```
# MAGIC cbt -instance=fieldeng-bigtable-demo-central lookup cc_fraud_bigtable 34687
# MAGIC ```
# MAGIC 
# MAGIC Get a specific row-if and column
# MAGIC 
# MAGIC ```
# MAGIC cbt -instance=fieldeng-bigtable-demo-central lookup cc_fraud_bigtable 9996 columns=pca001
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC delete all data
# MAGIC ```
# MAGIC cbt -instance=fieldeng-bigtable-demo-central deleteallrows cc_fraud_bigtable
# MAGIC ```

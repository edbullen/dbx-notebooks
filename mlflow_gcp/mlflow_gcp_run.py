# Databricks notebook source
# MAGIC %md
# MAGIC ## Test Batch Data

# COMMAND ----------

# DBTITLE 1,Create data-frame X_test_small of test data to score
import pandas as pd
data_dict = {
'id':[1,2,3,4,5,6,7,8,9,10]  ,
'pca[0]': [  -0.824643,   2.213661,   0.294360,   1.036710,  -2.824210, -12.224021,  -2.661802,  -2.150855,   0.857321,  -2.986466, ] ,
'pca[1]': [ -0.380659, -1.550982,  0.126646,  0.234517,  1.873888,  3.854150,  5.856393,  2.187917,  4.093912, -0.000891, ] ,
'pca[2]': [   1.406608,  -0.921553,   0.494768,   1.414188,  -2.604899, -12.466766,  -7.653616,  -3.430516,  -7.423894,   0.605887, ] ,
'pca[3]': [ -1.973115, -1.619521, -0.308863,  2.623882, -0.173365,  9.648311,  6.379742,  0.119476,  7.380245,  0.338338, ] ,
'pca[4]': [ -1.670249, -1.324002,  0.466604, -0.911082, -0.142998, -2.726961, -0.060712, -0.173210,  0.973366,  0.685448, ] ,
'pca[5]': [ -0.374518, -0.452981,  0.532789, -0.374297, -1.038574, -4.445610, -3.131550,  0.290700, -2.730762, -1.581954, ] ,
'pca[6]': [  -0.855799,  -1.228393,   0.509762,  -0.358195,   0.343429, -21.922811,  -3.103570,  -2.808988,  -1.496497,   0.504206, ] ,
'pca[7]': [  0.399257, -0.036892, -0.211086,  0.057491,  0.508249,  0.320792,  1.778492, -2.679351,  0.543015, -0.233403, ] ,
'pca[8]': [ -2.146896, -1.256747, -0.958940, -0.292507,  0.293139, -4.433162, -3.831154, -0.556685, -2.351190,  0.636768, ] ,
'pca[9]': [   0.800051,   1.745029,   0.649947,   0.635621,  -0.129888, -11.201400,  -7.191604,  -4.485483,  -3.944238,   1.010291, ] ,
'pca[10]': [ -1.395073,  0.288683, -1.824444, -0.357595, -1.166475,  9.328799,  7.102989,  1.903999,  6.355078,  0.004518, ] ,
'pca[11]': [  -1.678770,  -0.670951,  -0.232031,  -0.220115,   0.555617, -13.104933,  -9.928700,  -2.644219,  -7.309748,   0.044397, ] ,
'pca[12]': [ -0.228603, -0.573592,  1.193698, -0.342423,  1.553172,  0.888481, -0.067498, -0.982273,  0.748451,  0.420853, ] ,
'pca[13]': [  -0.171421,  -0.066173,  -0.523501,   0.100451,  -0.829716, -10.140200, -10.924187,  -4.691151,  -9.057993,  -0.931614, ] ,
'pca[14]': [  0.735139, -0.705275, -0.082841,  0.924730,  0.705358,  0.713465, -1.697914, -0.693080, -0.648945,  1.147974, ] ,
'pca[15]': [   0.106568,  -0.281511,  -1.519433,   1.042958,   0.484533, -10.098671,  -2.379421,  -2.553251,  -1.073117,   0.483063, ] ,
'pca[16]': [   0.427483,   0.244638,  -0.673627,  -0.677693,   0.524798, -17.506612,  -2.775114,  -3.483436,   1.524501,  -0.152471, ] ,
'pca[17]': [  0.641617,  0.500812,  1.602125,  0.024209,  0.802405, -8.061208,  0.273799, -0.064852,  1.831364, -0.285666, ] ,
'pca[18]': [  0.037305,  0.197147, -0.761916, -1.345256,  0.104247,  1.606870, -1.382188,  1.490329, -0.089724,  0.072550, ] ,
'pca[19]': [ -0.405119, -0.440636, -0.321995, -0.082952, -0.656172, -2.147181,  0.399097,  0.532145,  0.483303, -0.764274, ] ,
'pca[20]': [ -0.133007, -0.154222, -0.208393,  0.053064,  0.229827, -1.159830,  0.734775, -0.073205,  0.375026, -0.875146, ] ,
'pca[21]': [ -0.261955, -0.045589,  0.147181, -0.009799,  1.009158, -1.504119, -0.435901,  0.561496,  0.145400, -0.509849, ] ,
'pca[22]': [  -0.049717,   0.206068,   0.146409,   0.071704,  -0.144476, -19.254328,  -0.384766,  -0.075034,   0.240603,   1.313918, ] ,
'pca[23]': [ -0.174729, -0.444007, -0.026228,  0.671907, -0.672557,  0.544867, -0.286016, -0.437619, -0.234649,  0.355065, ] ,
'pca[24]': [ -0.330425, -0.282328, -1.169640,  0.138293, -0.246991, -4.781606,  1.007934,  0.353841, -1.004881,  0.448552, ] ,
'pca[25]': [ -0.320548, -0.194957, -0.820303, -0.097938, -0.309639, -0.007772,  0.413196, -0.521339,  0.435832,  0.193490, ] ,
'pca[26]': [ -0.063222,  0.003637,  0.178488,  0.014997, -1.995283,  3.052358,  0.280284,  0.144465,  0.618324,  1.214588, ] ,
'pca[27]': [  0.004724, -0.059951,  0.069996,  0.047826, -0.995793, -0.775036,  0.303937,  0.026588,  0.148469, -0.013923, ] ,
}
X_test_small = pd.DataFrame(data_dict)

# COMMAND ----------

X_test_small.columns

# COMMAND ----------

display(X_test_small.head(5))

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

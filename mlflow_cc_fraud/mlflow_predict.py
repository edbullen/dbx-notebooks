# Databricks notebook source
# MAGIC %md
# MAGIC # Predictions using ML Models managed by MLflow
# MAGIC Example scenarios demonstrated
# MAGIC 1. Get predictions using the **MLflow Rest API** (demonstrated in the previous "build" notebook)
# MAGIC 2. Apply a **Databricks MLflow model** using **SQL** against data queried from **BigQuery** - "batch scoring"
# MAGIC 3. Save results to **GCP BigTable** (HBase equivalent data-store)
# MAGIC 
# MAGIC *Many other options and approaches are possible*
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



# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. SQL Batch Score with Python UDF

# COMMAND ----------

# DBTITLE 1,Load the model and create a UDF function on it
import mlflow

# create a user-defined function that maps to MLflow model
fraud_detect_udf = mlflow.pyfunc.spark_udf(spark, "models:/cc_fraud/Production")
#Save the inference as a sql function that can be accessed in SPark / SQL
spark.udf.register("cc_fraud_model", fraud_detect_udf)

# COMMAND ----------

# DBTITLE 1,BigQuery Table of Demo Data to run Predictions against
# table ref is project.dataset.table
table = "fe-dev-sandbox.hsbc.cc_fraud_bq"

# COMMAND ----------

# DBTITLE 1,Sample Data Stored in BigQuery 
# load a sample of BigQuery Data
df_bq = spark.read.format("bigquery").option("table",table).load()
df_bq = df_bq.where(df_bq.id <= 5)
#df_bq.createOrReplaceTempView("bq_fraud")

# COMMAND ----------

display(df_bq.take(5))

# COMMAND ----------

# DBTITLE 1,Validation data to check the generated predictions - cc_fraud_valid
# MAGIC %sql
# MAGIC SELECT * FROM cc_fraud_valid LIMIT 5;

# COMMAND ----------

# DBTITLE 1,Example - Use the cc_fraud_model UDF to get predictions from data in a Databricks table
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

# DBTITLE 1,Extract a set of IDs from BigQuery to run Predictions on
# load a sample of BigQuery Data
df_bq = spark.read.format("bigquery").option("table",table).load()
df_bq = df_bq.where(df_bq.id <= 50000)
# create a view that we can query in SQL
df_bq.createOrReplaceTempView("bq_fraud")

# COMMAND ----------

# DBTITLE 1,Get Some Fraud Predictions
# MAGIC %sql 
# MAGIC SELECT `id`, `time`, `amountRange`, cc_fraud_model(
# MAGIC        `pca_0_`, `pca_1_`, `pca_2_`, `pca_3_`, `pca_4_`, `pca_5_`,
# MAGIC        `pca_6_`, `pca_7_`, `pca_8_`, `pca_9_`, `pca_10_`, `pca_11_`, `pca_12_`,
# MAGIC        `pca_13_`, `pca_14_`, `pca_15_`, `pca_16_`, `pca_17_`, `pca_18_`,
# MAGIC        `pca_19_`, `pca_20_`, `pca_21_`, `pca_22_`, `pca_23_`, `pca_24_`,
# MAGIC        `pca_25_`, `pca_26_`, `pca_27_`) as prediction
# MAGIC FROM bq_fraud
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Create a Temp SQL View of Predictions
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW cc_fraud_predictions
# MAGIC AS
# MAGIC SELECT `id`, `time`, `amountRange`, cc_fraud_model(
# MAGIC        `pca_0_`, `pca_1_`, `pca_2_`, `pca_3_`, `pca_4_`, `pca_5_`,
# MAGIC        `pca_6_`, `pca_7_`, `pca_8_`, `pca_9_`, `pca_10_`, `pca_11_`, `pca_12_`,
# MAGIC        `pca_13_`, `pca_14_`, `pca_15_`, `pca_16_`, `pca_17_`, `pca_18_`,
# MAGIC        `pca_19_`, `pca_20_`, `pca_21_`, `pca_22_`, `pca_23_`, `pca_24_`,
# MAGIC        `pca_25_`, `pca_26_`, `pca_27_`) as prediction
# MAGIC FROM bq_fraud;

# COMMAND ----------

# DBTITLE 1,Sample  Query of the Predictions from BQ table
# MAGIC %sql
# MAGIC SELECT `id`, `time`, `amountRange`, `label`
# MAGIC FROM cc_fraud_valid
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Sample Query of the Validation table, with "true" fraud labels
# MAGIC %sql
# MAGIC SELECT `id`, `time`, `amountRange`, `label` 
# MAGIC FROM cc_fraud_valid
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Join the Predictions made against the BQ data with the Validation data and check the accuracy
# MAGIC %sql
# MAGIC SELECT p.id, p.time, p.amountRange, p.prediction, v.label
# MAGIC FROM cc_fraud_predictions p
# MAGIC JOIN cc_fraud_valid v ON p.id = v.id
# MAGIC WHERE prediction != v.label;

# COMMAND ----------

# DBTITLE 1,Count the False Positives
# MAGIC %sql
# MAGIC SELECT count(p.id) as False_Postive_Count
# MAGIC FROM cc_fraud_predictions p
# MAGIC JOIN cc_fraud_valid v ON p.id = v.id
# MAGIC WHERE prediction != v.label
# MAGIC AND prediction == 1;

# COMMAND ----------

# DBTITLE 1,Example using SQL to get the predictions we are interested in 
# MAGIC %sql
# MAGIC SELECT `id`, `time`, `amountRange` FROM  cc_fraud_predictions 
# MAGIC WHERE amountRange > 6
# MAGIC AND prediction = 1;

# COMMAND ----------

# DBTITLE 1,Many Options for sourcing data and making model predictions available
# MAGIC %md
# MAGIC 
# MAGIC <img src="https://github.com/edbullen/dbx-notebooks/blob/136a93861baa721d0c2d996f7a27c979c0a2d32b/mlflow_gcp/images/HSBC_DDS_options.png?raw=true" alt="drawing" width="800"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Google BigTable
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

predictions_df.count()

# COMMAND ----------

predictions_df.show(5)

# COMMAND ----------

# DBTITLE 1,Write to BigTable in Iterative Loop
# Maximum number of mutations is 100000 per batch in the loop

import datetime
timestamp = datetime.datetime.utcnow()

feature_cols = ["time", "amountRange"]

column_family_1 = "cc_application"
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
    if counter >= batch_size-1:
        response = table.mutate_rows(hbase_row_ids)
        hbase_row_ids = []
        print(f"Wrote {counter+1} rows")
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
        
print(f"total records written {i+1}")    


# COMMAND ----------

predictions_df.where(predictions_df.prediction==1).take(5)

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
# MAGIC Get a row id=3  
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

predictions

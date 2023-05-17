# Databricks notebook source
# MAGIC %md
# MAGIC # Demo Prep
# MAGIC
# MAGIC Setup a sample Delta table for running test predictions against.  
# MAGIC
# MAGIC This is based on the Databricks sample data-sets:
# MAGIC
# MAGIC ```
# MAGIC /databricks-datasets/credit-card-fraud/data
# MAGIC ```

# COMMAND ----------

# set some vars to generate names for where we store our data
import re
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

# naming convention is <first>_<last>
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
# MAGIC # Source Data from Databricks Demo Datasets

# COMMAND ----------

path = '/databricks-datasets/credit-card-fraud/data/'
df = spark.read.format('parquet').options(header=True,inferSchema=True).load(path)

# COMMAND ----------

df.head()

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a Dedictated UC Catalog and Database (Schema)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS ${var.catalog_name}

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS ${var.catalog_name}.${var.db_name}

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Bronze Fraud Data Table

# COMMAND ----------

path = '/databricks-datasets/credit-card-fraud/data/'
df = spark.read.format('parquet').options(header=True,inferSchema=True).load(path)

# COMMAND ----------

# Extract the vector of PCA features into individual columns, Convert to a dataframe compatible with the pandas API
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
df_flattened = (df.withColumn("pca", vector_to_array("pcaVector"))).select(["time", "amountRange", "label"] + [col("pca")[i] for i in range(28)])

# COMMAND ----------

df_flattened.head()

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table ${var.catalog_name}.${var.db_name}.cc_fraud_bronze

# COMMAND ----------

df_flattened.write.mode("overwrite").saveAsTable("${var.catalog_name}.${var.db_name}.cc_fraud_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Silver Table with ID and load from Bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ${var.catalog_name}.${var.db_name}.cc_fraud_silver
# MAGIC (id bigint generated always as identity,
# MAGIC  time          int,
# MAGIC  amountRange	int,
# MAGIC  label			int,
# MAGIC  pca0			double,
# MAGIC  pca1			double,
# MAGIC  pca2			double,
# MAGIC  pca3			double,
# MAGIC  pca4			double,
# MAGIC  pca5			double,
# MAGIC  pca6			double,
# MAGIC  pca7			double,
# MAGIC  pca8			double,
# MAGIC  pca9			double,
# MAGIC  pca10			double,
# MAGIC  pca11			double,
# MAGIC  pca12			double,
# MAGIC  pca13			double,
# MAGIC  pca14			double,
# MAGIC  pca15			double,
# MAGIC  pca16			double,
# MAGIC  pca17			double,
# MAGIC  pca18			double,
# MAGIC  pca19			double,
# MAGIC  pca20			double,
# MAGIC  pca21			double,
# MAGIC  pca22			double,
# MAGIC  pca23			double,
# MAGIC  pca24			double,
# MAGIC  pca25			double,
# MAGIC  pca26			double,
# MAGIC  pca27          double
# MAGIC  );

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO ${var.catalog_name}.${var.db_name}.cc_fraud_silver
# MAGIC (time,
# MAGIC  amountRange,
# MAGIC  label,
# MAGIC  pca0,
# MAGIC  pca1,
# MAGIC  pca2,
# MAGIC  pca3,
# MAGIC  pca4,
# MAGIC  pca5,
# MAGIC  pca6,
# MAGIC  pca7,
# MAGIC  pca8,
# MAGIC  pca9,
# MAGIC  pca10,
# MAGIC  pca11,
# MAGIC  pca12,
# MAGIC  pca13,
# MAGIC  pca14,
# MAGIC  pca15,
# MAGIC  pca16,
# MAGIC  pca17,
# MAGIC  pca18,
# MAGIC  pca19,
# MAGIC  pca20,
# MAGIC  pca21,
# MAGIC  pca22,
# MAGIC  pca23,
# MAGIC  pca24,
# MAGIC  pca25,
# MAGIC  pca26,
# MAGIC  pca27)
# MAGIC SELECT * FROM ${var.catalog_name}.${var.db_name}.cc_fraud_bronze;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ${var.catalog_name}.${var.db_name}.cc_fraud_silver LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Gold Table with Predictions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ${var.catalog_name}.${var.db_name}.cc_fraud_gold
# MAGIC (id bigint ,
# MAGIC  time          int,
# MAGIC  amountRange	int,
# MAGIC  prediction int)

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleardown

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Either Drop or Delete from Gold Table

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM ${var.catalog_name}.${var.db_name}.cc_fraud_gold

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Drop Silver Table

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Drop Bronze Table

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Delete Model
# MAGIC
# MAGIC 1. Set from Production to None
# MAGIC 2. Delete the Model Version(s)
# MAGIC 3. Delete the Model

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 5. Delete the Experiment

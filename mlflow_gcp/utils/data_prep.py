# Databricks notebook source
# MAGIC %md 
# MAGIC # Helper Notebook to Prepare Data for Predictions Demo
# MAGIC 
# MAGIC Create two tables of data called 
# MAGIC ```
# MAGIC cc_fraud_test
# MAGIC ```
# MAGIC and
# MAGIC 
# MAGIC ```
# MAGIC cc_fraud_valid
# MAGIC ```
# MAGIC + The Test and Valid tables are Delta managed tables, both sourced from `/databricks-datasets//credit-card-fraud/data`  
# MAGIC + The pcaVector column has been exploded out to make a wide table with 28 PCA feature columns.  
# MAGIC + Both tables have a simulated transaction `id` column (matching for both tables)
# MAGIC + The Test table doesn't have the label column.  The Valid table has the label column in place for validating results predicted from data in the Test table

# COMMAND ----------

# MAGIC %sql
# MAGIC --DROP TABLE IF EXISTS cc_fraud_load;
# MAGIC DROP TABLE IF EXISTS cc_fraud_test;
# MAGIC DROP TABLE IF EXISTS cc_fraud_valid;

# COMMAND ----------

# DBTITLE 1,Load into Spark DF for pre-processing
# MAGIC %python
# MAGIC df = spark.read.format('parquet').options(header=True,inferSchema=True).load('/databricks-datasets/credit-card-fraud/data/')
# MAGIC 
# MAGIC from pyspark.ml.functions import vector_to_array
# MAGIC from pyspark.sql.functions import col
# MAGIC 
# MAGIC # extract the vector of PCA features into individual columns in a Pandas data-frame - not always necessary to do this, just doing for simple demo
# MAGIC df_wide = (df.withColumn("pca", vector_to_array("pcaVector"))).select(["time", "amountRange", "label"] + [col("pca")[i] for i in range(28)])

# COMMAND ----------

# DBTITLE 1,Add an ID col
from pyspark.sql.functions import monotonically_increasing_id

# This will return a new DF with all the columns + id
df_wide = df_wide.withColumn("id", monotonically_increasing_id())

# COMMAND ----------

# DBTITLE 1,Re-Order the columns
cols = df_wide.columns

# move id col to start
new_cols_tmp = [cols.pop()]
new_cols_tmp.extend(cols)

# move the label col to the end
new_cols = new_cols_tmp[0:3]
new_cols.extend(new_cols_tmp[4:])
new_cols.extend(new_cols_tmp[3:4])


df_new_cols = df_wide.select(*new_cols)

# COMMAND ----------

display(df_new_cols.take(5))

# COMMAND ----------

# DBTITLE 1,Write out to Delta Table cc_fraud_valid
df_new_cols.write.format("delta").saveAsTable("default.cc_fraud_valid")

# COMMAND ----------

# DBTITLE 1,Write out to Delta Table cc_fraud_test
# strip off the label col
df_test = df_new_cols.drop('label')

df_test.write.format("delta").saveAsTable("default.cc_fraud_test")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from cc_fraud_valid limit 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from cc_fraud_test limit 10;

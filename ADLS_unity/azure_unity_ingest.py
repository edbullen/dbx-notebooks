# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### Azure Data Lake Storage (ADLS) and Databricks
# MAGIC
# MAGIC <img src="https://oneenvstorage.blob.core.windows.net/images/adlslogo.png" width="400">
# MAGIC
# MAGIC | Specs                |                                   |
# MAGIC |----------------------|-----------------------------------|
# MAGIC | Azure Resource Group | oneenv                            |
# MAGIC | ADLS Account         | oneenvadls                        |
# MAGIC | Containers           | deltalake, files                  |
# MAGIC | Region               | US West                           | 
# MAGIC
# MAGIC <br />
# MAGIC
# MAGIC 1. Use SAS Key
# MAGIC 2. Use the Azure Key Vault backed secrets scope
# MAGIC
# MAGIC Docs: https://docs.databricks.com/data/data-sources/azure/azure-datalake-gen2.html

# COMMAND ----------

#dbutils.secrets.list('oetrta')

# COMMAND ----------

#dbutils.secrets.listScopes()

# COMMAND ----------

# MAGIC %md
# MAGIC **DBX Workspace**: `field-eng-east`  
# MAGIC **Storage Account** : `oneenvadls`  
# MAGIC **Resource Group**: `oneenv`  
# MAGIC **Storage Container**: `deltalake`  
# MAGIC
# MAGIC Path to my data:  
# MAGIC `abfss://<storage-container>@<storage-account>.dfs.core.windows.net`
# MAGIC
# MAGIC Path to some sample data:
# MAGIC `deltalake/delta/scott/nyc_taxi`
# MAGIC

# COMMAND ----------

# The below details are related to the Service Principal oneenv-adls
APPLICATION_ID = "ed573937-9c53-4ed6-b016-929e765443eb"
DIRECTORY_ID = "9f37a392-f0ae-4280-9796-f1864a10effc"
APP_KEY = dbutils.secrets.get(scope = "oetrta", key = "oneenv-adls-secret")

# COMMAND ----------

#configs = {"fs.azure.account.auth.type": "OAuth",
#           "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
#           "fs.azure.account.oauth2.client.id": APPLICATION_ID,
#           "fs.azure.account.oauth2.client.secret": APP_KEY,
#           "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/"+DIRECTORY_ID+"/oauth2/token"}
          

# COMMAND ----------

spark.conf.set("fs.azure.account.auth.type", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id", APPLICATION_ID)
spark.conf.set("fs.azure.account.oauth2.client.secret", APP_KEY)
spark.conf.set("fs.azure.account.oauth2.client.endpoint", "https://login.microsoftonline.com/"+DIRECTORY_ID+"/oauth2/token")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unity Catalog Storage Credential
# MAGIC Before Creating an External Location, a Unity Catalog Storage Credential needs to be set up.
# MAGIC
# MAGIC 1. In Azure, register an *Identity App* (see "App Registrations" in the console)
# MAGIC <img src="" width="400">
# MAGIC
# MAGIC https://drive.google.com/file/d/1D7eWjLnBVzYksKCW_InGh_jKasgGzx-J/view?usp=sharing
# MAGIC https://drive.google.com/file/d/1D7eWjLnBVzYksKCW_InGh_jKasgGzx-J/view?usp=sharing
# MAGIC
# MAGIC 2. In Databricks Unity Catalog Data Explorer, Create an *External Data Storage Credential* that links to the App ID and stores a Key associated with the the registered Azure Indentity App. 

# COMMAND ----------

# DBTITLE 1,Create a Unity Catalog External Location - ref's a UC Storage Credential
# MAGIC %sql
# MAGIC CREATE EXTERNAL LOCATION nyc_taxi
# MAGIC URL 'abfss://deltalake@oneenvadls.dfs.core.windows.net/delta/scott/nyc_taxi'
# MAGIC WITH (STORAGE CREDENTIAL field_demos_credential);

# COMMAND ----------

# DBTITLE 1,Create a Unity Catalog External Volume - path is set up as UC External Location
# MAGIC %sql
# MAGIC CREATE EXTERNAL VOLUME users.ed_bullen.nyc_taxi
# MAGIC     LOCATION 'abfss://deltalake@oneenvadls.dfs.core.windows.net/delta/scott/nyc_taxi'
# MAGIC     COMMENT 'Unity External Volume for NYC Taxi Data staged as Parquet files in Azure ADLS';

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE EXTERNAL TABLE users.ed_bullen.nyc_taxi LOCATION 'abfss://deltalake@oneenvadls.dfs.core.windows.net/delta/scott/nyc_taxi';

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL users.ed_bullen.wiki_articles;

# COMMAND ----------



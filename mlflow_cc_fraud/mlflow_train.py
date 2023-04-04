# Databricks notebook source
# MAGIC %md
# MAGIC # Train and Deploy ML Models with MLFlow and Databricks
# MAGIC 
# MAGIC Simple demonstration using a publicly available retail banking data-set to train a *Credit Card Fraud Detection* model, based on a Random Forest decision tree.  
# MAGIC 
# MAGIC Make the model available for multiple access paths via Pyspark and SQL in the data-layer via Databricks and also via a Rest API using MLflow.
# MAGIC + Batch scoring - query in SQL via a Spark UDF  
# MAGIC + Model serving - serve out of the MLflow Rest API directly for real-time scoring
# MAGIC + Data pipeline - write results out Delta to table.
# MAGIC 
# MAGIC Demonstrate MLflow benefits integrated with Databricks and Delta Lake
# MAGIC + ML development environment integrated with the data-platform (ease of use and simplicity)
# MAGIC + Track ML experiments for collaboration (explainability and governance)
# MAGIC + Programaticaly choose the best experiment and deploy to the Model Registry (automation)
# MAGIC + Manage lifecycle of model versions and deployment lifecycle (MLOps) 
# MAGIC + Governance and Explainability of models stored with artifacts mapped to deployment-versions
# MAGIC 
# MAGIC ## ML-Ops
# MAGIC .  
# MAGIC    
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1snUQ1VE0pV5rxlbjpH1257hYCAwdqZVU" alt="drawing" width="600"/>
# MAGIC 
# MAGIC ## Demo Structure
# MAGIC 
# MAGIC This demonstration is split into two parts with two separate notebooks:
# MAGIC 1. **Build**: Train, Validate and Deploy (this notebook)  
# MAGIC   a) load a data-set and train a model  
# MAGIC   b) multiple training runs tracked in MLflow ("experiments")    
# MAGIC   c) store artefacts with each model version for explainablility and governance.     
# MAGIC 2. **Run**: Test using the model via different interfaces (notebook 2)  
# MAGIC   a) Batch-score results via Databricks SQL with a UDF  
# MAGIC   b) Get predictions from the Databricks MLflow Rest API  
# MAGIC   c) Batch Score and write results to a Delta table.  
# MAGIC 
# MAGIC The model is built as a Scikit-learn Random Forest model, managed in the MLflow framework and deployed in Databricks so that it is integrated into the Lakehouse environment.  
# MAGIC   
# MAGIC This means features can be read in and applied to the model from Lakehouse data-sources and written out to Lakehouse SQL tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks versions and MLflow library
# MAGIC Ensure the MLflow Python libraries are installed on the cluster or use a Databricks "ML" cluster type. 
# MAGIC 
# MAGIC Tested with versions:
# MAGIC + DBR LTS 10.4 Spark 3.2.1 
# MAGIC + DBR LTS 11.3 Spark 3.3.0 
# MAGIC + DBR LTS 12.2 Spark 3.3.2
# MAGIC  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Set Description
# MAGIC 
# MAGIC ref. Kaggle - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 
# MAGIC 
# MAGIC The dataset contains transactions made by credit cards in September 2013 by European cardholders.
# MAGIC This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# MAGIC 
# MAGIC It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. 
# MAGIC + Features **V1, V2, … V28** are the principal components obtained with PCA
# MAGIC + the only features which have not been transformed with PCA are 'Time' and 'Amount'.  
# MAGIC + Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. 
# MAGIC + The feature 'Amount' is the transaction amount. 
# MAGIC + Feature **'Class'** is the response variable and it takes value **1** in case of fraud and **0** otherwise.

# COMMAND ----------

# DBTITLE 1,Raw Data is in Delta Format
display(dbutils.fs.ls('/databricks-datasets//credit-card-fraud/data'))

# COMMAND ----------

# MAGIC %pip install databricks-feature-store

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Load the Data and Explore
# MAGIC Load into Pyspark.  Use the [Spark Pandas library](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html) to use Pandas API syntax but with data and processing distributed in a cluster.

# COMMAND ----------

# DBTITLE 1,Load Credit Card Data Set into a Spark Dataframe
path = '/databricks-datasets/credit-card-fraud/data/'
df = spark.read.format('parquet').options(header=True,inferSchema=True).load(path)

# COMMAND ----------

# DBTITLE 1,Load into a Spark Pandas dataframe 

from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
import databricks.feature_store

from databricks.feature_store import feature_table
# Convert to a dataframe compatible with the pandas API
import pyspark.pandas as ps

# Extract the vector of PCA features into individual columns, Convert to a dataframe compatible with the pandas API
spark_pandas = (df.withColumn("pca", vector_to_array("pcaVector"))).select(["time", "amountRange", "label"] + [col("pca")[i] for i in range(28)]).pandas_api()

# COMMAND ----------

# DBTITLE 1,View First 5 rows of Dataframe
spark_pandas.loc[0:5].iloc[:,2:13]

# COMMAND ----------

# DBTITLE 1,Visualise Data with Pairplots
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#take a sample to speed things up - only first 100 rows, just 6 cols of features
sample_mixed_df = spark_pandas.head(100).iloc[:,2:9]
# because data is skewed, i.e. mainly not fraud, need to explicitly sample some frauds to help the visualisation
sample_fraud_df = spark_pandas[spark_pandas.label==1].head(50).iloc[:,2:9]
# add the 100 row sample and 50 rows of explicit fraud data together
sample_df = ps.concat([sample_mixed_df, sample_fraud_df])

# convert this sample to real Pandas to work with some visualisation libraries 
pandas_df = sample_df.to_pandas()
sns.set(font_scale=2.0)
sns.pairplot( pandas_df
             , vars=pandas_df.iloc[:,1:9]
             , hue="label"
             , height=2)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the Data for Training a Model
# MAGIC 
# MAGIC Follow a typical basic Machine Learning model approach:
# MAGIC + Split the data up into Features (`X`) and Labels (`y`)  
# MAGIC + Randomly Split the Features and Labels into Training Data and Test Data for testing the models performance. 
# MAGIC + Train the data on the training data-set and check the performance with the test set. 
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1Ed7UPfc8i9AE_uERKPcMT9015QkMe60q" alt="drawing" width="800"/>

# COMMAND ----------

sqlContext.setConf("spark.sql.shuffle.partitions", "8")

# COMMAND ----------

# DBTITLE 1,Split into Test and Training Data-Sets
# in this example we missed off the first two features - time and amount-range. 
# This is just for simplicity for an easy demo - probably they are good features to have

# convert Pandas Pyspark to regular Spark so we can use native Spark random split function
train, test = spark_pandas.to_spark().randomSplit(weights=[0.8,0.2], seed=200)

# Split the train and test data into X features and y predictors, and convert back to Spark Pandas
X_train = train.pandas_api().iloc[:,3:]
X_test = test.pandas_api().iloc[:,3:]

y_train = train.pandas_api().iloc[:,2]
y_test = test.pandas_api().iloc[:,2]


# COMMAND ----------

# DBTITLE 1,Check the Shape / Dimensions of the Train and Test data
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# COMMAND ----------

# DBTITLE 1,Optional - examine the Training Data
X_train.head()

# COMMAND ----------

# DBTITLE 1,Optional - examine the Training Data
y_train.head()

# COMMAND ----------

X_train['pca[0]'].head()

# COMMAND ----------

# DBTITLE 1,Optional - confirm the training data type (Pandas API on Spark)
print(type(X_train))
print(type(y_train))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a Model in MLflow
# MAGIC 
# MAGIC 
# MAGIC #### Supervised Learning process
# MAGIC Labeled data is needed to train a model to generate label predictions.  
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1--qOV9nfiesXB_EwQKdToLqIIRyCm8bQ" alt="drawing" width="600"/>
# MAGIC 
# MAGIC 
# MAGIC ### Random Forest Classification Algorithm
# MAGIC 
# MAGIC This notebook demostrated using a simple **Random Forest** algorithm to predict Fraud (label=1) vs Not-Fraud (label=0) based on the training data-set.  Random Forest is a Supervised ML algorithm.  
# MAGIC 
# MAGIC MLflow and Databricks support many other more advanced ML algorithms including
# MAGIC + MLlib
# MAGIC + Tensorflow  
# MAGIC + Keras  
# MAGIC + Scikit-learn  
# MAGIC + XGBoost  
# MAGIC + Spacy
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1-9gDhtXQQXizghiQZz7K38lC3VVg0WUs" alt="drawing" width="550"/>
# MAGIC 
# MAGIC Final result determined by majority vote or average.

# COMMAND ----------

# MAGIC %md  
# MAGIC 
# MAGIC Two types of Random Forest model:  
# MAGIC `RandomForestRegressor` - this gives a percentage likelihood of 1 or 0.  It also takes longer to train.  
# MAGIC `RandomForestClassifier` - this gives a 1 / 0 indicator of Fraud / Not-Fraud

# COMMAND ----------

# MAGIC %md
# MAGIC ### Accuracy, Precision, Recall
# MAGIC 
# MAGIC *Precision* (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, while *recall* (also known as sensitivity) is the fraction of relevant instances that were retrieved. Both precision and recall are therefore based on relevance.
# MAGIC   
# MAGIC + Low Recall score - due to many missed Frauds  
# MAGIC + High Precision - few false positives 
# MAGIC 
# MAGIC These two measures can be combined into a single **F1 Score** to measure how accurate the model is.
# MAGIC 
# MAGIC 
# MAGIC <!-- <img src="https://drive.google.com/uc?export=view&id=1-6Pbged7EQlCQsSzmyKYw8ApN97kcNg1" alt="drawing" width="400"/> -->

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multiple Model Training Experiments in MLFlow
# MAGIC .   
# MAGIC    
# MAGIC + *Experiments* - track all our machine-learning development runs and associated artefects
# MAGIC + *Feature-Store* - manage the data we use to train the model; retain this for model reproducibility and team collaboration (*this has been missed out of this demo*)
# MAGIC + *Models* - stored in the Databricks environment with namespace mapped to release cycle deployment

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC If we don't set the MLflow experiment name, it is set to the notebook name.
# MAGIC   
# MAGIC Either give each MLflow training run a unique name, or let it generate the names for you.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Libraries and set up Demo Paths

# COMMAND ----------

# DBTITLE 1,Import MLflow library
import mlflow
print(mlflow.__version__)

# COMMAND ----------

# DBTITLE 1,Setup Variables for Demo paths and data-store location
# set some vars to generate names for where we store our experiments and feature data
import re
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

db_name = current_user_no_at

print(f"db_name set to {db_name}")
print(f"current_user is {current_user}")
print(f"Log experiments to /Users/{current_user}/mflow_fraud_simple")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set MLFlow Experiment Location

# COMMAND ----------

mlflow.set_experiment(f"/Users/{current_user}/mflow_fraud_simple")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train the Model with GridSearch for Hyperparameter Optimisations

# COMMAND ----------

# DBTITLE 1,Local Pandas DF training data
# Either use Spark MLlib, or copy a set of training data into the Spark driver node and use regualar Pandas with SciKit
X_train_pd = X_train.to_pandas()
y_train_pd = y_train.to_pandas()
X_test_pd = X_test.to_pandas()
y_test_pd = y_test.to_pandas()
print("Training Data - Size in KBytes", X_train_pd.memory_usage(index=True).sum()/1024)

# COMMAND ----------

# DBTITLE 1,MLFlow model train - takes 2 minutes
# import Scikit-Learn open source ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# MLflow library for recording the model signature
from mlflow.models.signature import infer_signature

# open source python libs
import seaborn as sns
import numpy as np

import time

# set a maximum number of branches (speed up training time)
max_depth = [5, 10, 20]

# assess the impact of different number of trees
estimators = [10, 20, 50]

# timestamp to append to the run-name for this set of runs
run_timestamp = int(time.time())

for depth in max_depth:
    for n_estimators in estimators:
      with mlflow.start_run(run_name=f'cc_fraud_{n_estimators}_{run_timestamp}') as run:
        # Auto-logging to log metrics, parameters, and models without explicit log statements. 
        #   silent = suppress event logs and warnings
        #mlflow.sklearn.autolog(silent=True)

        # Set the hyperparameters for the model - i.e. estimators is the number of trees to use
        model_rf = RandomForestClassifier(n_estimators = n_estimators
                                         , random_state = 42
                                         , max_depth = depth
                                         , n_jobs = 16)
        # Train the model on training data
        model_rf.fit(X_train_pd, y_train_pd);

        # Calculate accuracy metrics
        y_pred_pd = model_rf.predict(X_test_pd).astype(int)

        accuracy = accuracy_score(y_test_pd, y_pred_pd)*100
        prec = precision_score(y_test_pd, y_pred_pd)*100
        rec = recall_score(y_test_pd, y_pred_pd)*100
        f1 = f1_score(y_test_pd, y_pred_pd)*100

        # Set a tag so we can find this group of experiment runs
        mlflow.set_tag("project", "cc_fraud")

        # Infer the model signature before we log it
        signature = infer_signature(X_train_pd, model_rf.predict(X_train_pd))

        # Log model called "cc_fraud" in the MLflow repository with parameters, and metrics
        mlflow.sklearn.log_model(model_rf
                                 , "cc_fraud"
                                 , input_example=X_train_pd.loc[1:1]
                                 , serialization_format = 'pickle'
                                 , signature=signature
                                 )

        # Store the accuracy metrics in MLflow for this training run 
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', prec)
        mlflow.log_metric('recall', rec)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_param('max_depth', depth)
        mlflow.log_param('n_estimators', n_estimators)

        # Record a confusion matrix plot for the model test results
        fig, ax = plt.subplots(figsize=(6, 6))  # set dimensions of the plot
        CM = confusion_matrix(y_test_pd, y_pred_pd)
        ax = sns.heatmap(CM, annot=True, cmap='Blues', fmt='.8g')
        mlflow.log_figure(plt.gcf(), "test_confusion_matrix.png")

        # Store a feature-importance plot as an artefact
        importances = model_rf.feature_importances_.round(3)
        sorted_indices = np.argsort(importances)[::-1]
        feat_labels = X_train_pd.columns
        fig, ax = plt.subplots(figsize=(8, 8))    # set dimensions of the plot
        
        sns.barplot(x=importances[sorted_indices]  # plot the relative importance of features
                    , y=feat_labels[sorted_indices]
                    , orient='horizonal')
        for i in ax.containers:
            ax.bar_label(i,)
        mlflow.log_figure(plt.gcf(), "feature_importance.png")    

 
mlflow.end_run()  



# COMMAND ----------

# DBTITLE 1,Get the Best Model
#get the best model from the runs in our experiment
best_model = mlflow.search_runs(filter_string='tags.project = "cc_fraud"', order_by = ['metrics.f1_score DESC']).iloc[0]

# COMMAND ----------

best_model

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Deploy Model to MLflow

# COMMAND ----------

# DBTITLE 1,MLFlow Register Model
model_registered = mlflow.register_model("runs:/"+best_model.run_id+"/cc_fraud", "cc_fraud")

# COMMAND ----------

# DBTITLE 1,Transition Model to Production - either via UI or Programatically
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage("cc_fraud", model_registered.version, stage = "Production", archive_existing_versions=True)
print("model version "+model_registered.version+" as been registered as production ready")

# COMMAND ----------

# MAGIC %md
# MAGIC **Model Namespace**
# MAGIC ```
# MAGIC                       stage or version
# MAGIC          model name     |
# MAGIC             |           |
# MAGIC "models:/cc_fraud/Production"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Model to MLflow Rest API
# MAGIC The MLflow model serving API allows predictions to be scored via a secured API.  This makes the model easy to integrate with any code or software that can make a REST API call.
# MAGIC 
# MAGIC The new Serverless MLFlow Serving API takes about 5 minutes to deploy. 
# MAGIC .  
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1-7CsjhK0cFj96yErtCd2VWIJf0ypNHHJ" alt="drawing" width="1200"/>

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLFlow REST API Call Example
# MAGIC 
# MAGIC Use a Python IDE (Pycharm or Visual Studio) to call the REST API with some sample data.  
# MAGIC   
# MAGIC Payload is provided in this format: 
# MAGIC ```
# MAGIC {
# MAGIC     "dataframe_records": [
# MAGIC         {
# MAGIC             "pca[0]": 1.226073,
# MAGIC             "pca[1]": -1.640026,
# MAGIC             ...
# MAGIC             ...
# MAGIC             "pca[27]": 0.012658,
# MAGIC         }
# MAGIC     ]
# MAGIC }
# MAGIC            
# MAGIC ```
# MAGIC Results are returned as a list of predictions:
# MAGIC ```
# MAGIC {'predictions': [0, 1, 0]}
# MAGIC ```

# COMMAND ----------

# MAGIC %sh mlflow --version

# COMMAND ----------

# MAGIC %md
# MAGIC **Model Namespace**
# MAGIC ```
# MAGIC                       stage or version
# MAGIC          model name     |
# MAGIC             |           |
# MAGIC "models:/cc_fraud/Production"
# MAGIC ```

# COMMAND ----------

# Model URI is the location of the model in MLFLow
model_uri="models:/cc_fraud/Production"

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix - Formatting test data
# MAGIC Generate inline data from original data-source

# COMMAND ----------

import pandas as pd

# Take a sample of test data
X_test_small = X_test.head(5)
print(round(X_test_small.memory_usage(index=True, deep=True).sum()/1024), "KB")

# COMMAND ----------

# X_test_small_2 has true fraud values
mask = y==1
X_test_small_2=X_test[mask].head(5)
X_test_small = X_test_small.append(X_test_small_2)

# COMMAND ----------

import re
dynamic_text = []
for i in range(0,28):
  col = X_test_small[f'pca[{i}]'].to_string(index=False)
  flat_col = re.sub("\n", ", ", col)
  dynamic_text.append(flat_col + ',')

# COMMAND ----------

print("data_dict = {")
for i,t in enumerate(dynamic_text):
  print(f'\'pca[{i}]\': [', t, ']', ',' )
print("}")  

# Databricks notebook source
# MAGIC %md
# MAGIC # End to End train and deploy ML Model with MLFlow and Vertex AI
# MAGIC 
# MAGIC Demonstration using a publicly available retail banking data-set to train a *Credit Card Fraud Detection* model.  
# MAGIC 
# MAGIC Make the model available for multiple access paths via Pyspark, SQL and MLflow and integrate with GCP:
# MAGIC + Batch scoring - query in SQL via a Spark UDF  
# MAGIC + Serve out of the MLflow Rest API directly for real-time scoring
# MAGIC + Write results out to GCP BigTable.
# MAGIC + Deploy to GCP Vertex AI, access via the Vertex.ai endpoint
# MAGIC 
# MAGIC Demonstrate MLflow benefits integrated with Databricks and Delta Lake
# MAGIC + ML development environment integrated with the data-platform (ease of use and simplicity)
# MAGIC + Track ML experiments for collaboration (explainability and governance)
# MAGIC + Programaticaly choose the best experiment and deploy to the Model Registry
# MAGIC + Manage lifecycle of model versions and deployment lifecycle (Dev -> Staging -> Prod) 
# MAGIC + Governance and Explainability of models stored with artifacts mapped to deployment-versions
# MAGIC 
# MAGIC ## ML-Ops
# MAGIC .  
# MAGIC    
# MAGIC    
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1snUQ1VE0pV5rxlbjpH1257hYCAwdqZVU" alt="drawing" width="600"/>
# MAGIC 
# MAGIC ## Demo Structure
# MAGIC 
# MAGIC This Demo is split into two parts with two separate notebooks:
# MAGIC 1. **Build**: Train, Validate and Deploy (this notebook)  
# MAGIC   a) load a data-set and train a model, including a basic ML 101 introduction  
# MAGIC   b) demonstrate multiple training runs tracked in MLflow ("experiments").  
# MAGIC   c) store artefacts with each model version for explainablility and governance.     
# MAGIC 2. **Run**: Test using the model via different interfaces (notebook 2)  
# MAGIC   a) Batch-score results via Databricks SQL with UDF  
# MAGIC   b) Batch Score and write results to GCP Hbase (BigTable)  
# MAGIC   c) Access via Vertex AI (GCP) endpoint. 
# MAGIC 
# MAGIC The model is built as a Scikit-learn Random Forest model, managed in the MLflow framework and deployed in Databricks for using directly against the data-lake as well as into Vertex.AI for serving.
# MAGIC 
# MAGIC Service Account setup:
# MAGIC 
# MAGIC + Privileges for Vertex: `Storage Admin`, `Vertex AI Administrator`   
# MAGIC + Privileges for HBASE: `BigTable User`  
# MAGIC 
# MAGIC Blog refs: https://www.databricks.com/blog/2022/08/12/mlops-on-databricks-with-vertex-ai-on-google-cloud.html 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparation
# MAGIC Install the correct libraries and library versions for MLflow and the Google MLflow connector
# MAGIC - `pip install mlflow==1.26.1` or install it as a library in the cluster 
# MAGIC - `pip install google_cloud_mlflow` or install it as a library in the cluster (tested with version `0.0.6`)
# MAGIC 
# MAGIC Next, load the data for building and testing the machine learning model.  
# MAGIC 
# MAGIC Tested with versions:
# MAGIC + DBR LTS 10.4 Spark 3.2.1 
# MAGIC + DBR LTS 11.3 Spark 3.3.0 
# MAGIC   - Vertex AI Real-time Scoring issue : 'google.protobuf.pyext._message.RepeatedCompositeCo' object has no attribute 'WhichOneof'

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
# MAGIC + Feature **'Class'** is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# COMMAND ----------

display(dbutils.fs.ls('/databricks-datasets//credit-card-fraud/'))

# COMMAND ----------

# DBTITLE 1,Raw Data is in Delta Format
display(dbutils.fs.ls('/databricks-datasets//credit-card-fraud/data'))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Load the Data and Explore
# MAGIC Load into Pyspark.  Then convert to Pandas for convenience (this means the data-set has to reside in memory on one node)

# COMMAND ----------

# DBTITLE 1,Load Credit Card Data Set into a Spark Dataframe
path = '/databricks-datasets//credit-card-fraud/data/'
df = spark.read.format('parquet').options(header=True,inferSchema=True).load(path)

# COMMAND ----------

display(df.head(5))

# COMMAND ----------

df.count()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# DBTITLE 1,Load into Pandas dataframe (load into driver memory)
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

# extract the vector of PCA features into individual columns in a Pandas data-frame - not always necessary to do this, just doing for simple demo
pandas_df = (df.withColumn("pca", vector_to_array("pcaVector"))).select(["time", "amountRange", "label"] + [col("pca")[i] for i in range(28)]).toPandas()

# COMMAND ----------

pandas_df.loc[0:5]

# COMMAND ----------

# DBTITLE 1,Check how much memory this takes up on the Driver Node
print(round(pandas_df.memory_usage(index=True, deep=True).sum()/1024/1024), "MB")

# COMMAND ----------

# DBTITLE 1,Data is very skewed - Random Forest is still robust in this situation 
#pandas_df.groupby(['label']).count()
pandas_df.value_counts(subset=['label']) 

# COMMAND ----------

# DBTITLE 1,Visualise Data
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#take a sample to speed things up - only 100 rows, just 10 cols of features
sample_mixed_df = pandas_df.sample(100).iloc[:,2:13]

# because data is skewed, i.e. mainly not fraud, need to explicitly sample some fruads to help the visualisation
sample_fraud_df = pandas_df[pandas_df.label==1].sample(50).iloc[:,2:13]

sample_df = pd.concat([sample_mixed_df, sample_fraud_df])

sns.pairplot( sample_df
             , vars=sample_df.iloc[:,3:13]
             , hue="label")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the Data for Training a Model
# MAGIC 
# MAGIC Follow a 101 Machine Learning model approach:
# MAGIC + Split the data up into Features (`X`) and Labels (`y`)  
# MAGIC + Randomly Split the Features and Labels into Training Data and Test Data for testing the models performance. 
# MAGIC + Train the data on the training data-set and check the performance with the test set. 

# COMMAND ----------

# DBTITLE 1,Divide dataframe into X features and y Labels
# in this example we missed off the first two features - time and amount-range. This is just for simplicity - probably they are good features
X = pandas_df.iloc[:,3:]
y = pandas_df.iloc[:,2]

# COMMAND ----------

print("Distinct y-label values:", y.unique())
print("\nSample of X values:\n", X.iloc[:1,0:7])

# COMMAND ----------

# DBTITLE 1,Split into Test and Training Data-Sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# COMMAND ----------

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# COMMAND ----------

X_train.head()

# COMMAND ----------

y_train.head()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a Model - Vanilla
# MAGIC 
# MAGIC ### Basic Concepts in Machine Learning
# MAGIC 
# MAGIC In general we either train a model based on past known events - this is known as **Supervised Learning**.  Supervised Learning can fall into categories of either 
# MAGIC + *Classification* (is this a picture of a cat or a dog, is this a fraud or not a fraud)  
# MAGIC + *Regression* (how much will this stock be worth in the future, what is the predicited weather temperature range)
# MAGIC 
# MAGIC It is also possible to create machine learning algorithms that don't require training on previous known outcomes.  This is known as **Unsupervised Learning**.  Most supervised learning is achieved by various
# MAGIC + *Clustering* algorithms (identify different related groups of data, which data-points can we consider anomolies)
# MAGIC 
# MAGIC #### Supervised Learning process
# MAGIC Labeled data is needed to train a model to generate label predictions.  
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1--qOV9nfiesXB_EwQKdToLqIIRyCm8bQ" alt="drawing" width="400"/>
# MAGIC 
# MAGIC #### Unsupervised Learning process
# MAGIC No training against pre-labeled data is needed.  
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1-7eB7cfmQwLAuHEJ9KhE9PIs3W31UmfW" alt="drawing" width="400"/>
# MAGIC 
# MAGIC ### Random Forest Classification Algorithm
# MAGIC 
# MAGIC This notebook demostrated using a **Random Forest** algorithm to predict Fraud (label=1) vs Not-Fraud (label=0) based on the training data-set.  Random Forest is a Supervised ML algorithm.  
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1-9gDhtXQQXizghiQZz7K38lC3VVg0WUs" alt="drawing" width="400"/>
# MAGIC 
# MAGIC Final result determined by majority vote or average.

# COMMAND ----------

# MAGIC %md
# MAGIC   
# MAGIC Optionally, skip forward to Train a Model - MLFlow
# MAGIC 
# MAGIC `RandomForestRegressor` - this gives a percentage likelihood of 1 or 0.  It also takes longer to train.  
# MAGIC `RandomForestClassifier` - this gives a 1 / 0 indicator of Fraud / Not-Fraud

# COMMAND ----------

# DBTITLE 1,Vanilla non-MLFlow way
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Set the hyperparameters for the model - i.e. estimators is the number of trees to use. More trees = more training time
model_rf = RandomForestClassifier(n_estimators=5, max_depth=10, random_state=42, n_jobs=16)
# Train the model on training data
model_rf.fit(X_train, y_train);

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding the Model

# COMMAND ----------

# DBTITLE 1,Each decision tree estimator 0..n can be viewed
# https://towardsdatascience.com/scikit-learn-decision-trees-explained-803f3812290d
from sklearn import tree
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (50,30), dpi=300)
tree.plot_tree(model_rf.estimators_[4], fontsize=5)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The left node is True and the right node is False.  
# MAGIC The value line in each box is telling you how many samples at that node fall into each category, in order.
# MAGIC 
# MAGIC   
# MAGIC Here is a zoomed in view of the top part of one of the trees:
# MAGIC 
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1-Qchz3_BDdwFldRoQX1Tj2CwvX70XK4u" alt="drawing" width="1200"/>

# COMMAND ----------

# DBTITLE 1,Feature Importance
import seaborn as sns
import numpy as np

# get the pct contribution to the result
importances = model_rf.feature_importances_.round(3)

# Sort the feature importance in descending order
sorted_indices = np.argsort(importances)[::-1]
feat_labels = X_train.columns

# setting the dimensions of the plot
fig, ax = plt.subplots(figsize=(10, 10))

# plot the relative importance of features
sns.barplot(x=importances[sorted_indices]
            , y=feat_labels[sorted_indices]
            , orient='horizonal')

for i in ax.containers:
    ax.bar_label(i,)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assess the Performance of the Model
# MAGIC Use the Test data-set to generate preditions.  
# MAGIC Compare the model output with the actual `y_test` values that were split off from the `X_test` data-set

# COMMAND ----------

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# get some predictions on our test features, use these to compare with actual y_test vals. Convert to type int.
y_pred = model_rf.predict(X_test).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# COMMAND ----------

print("Accuracy", round(acc,5))
print("Precision", round(prec,5))
print("Recall", round(rec,5))
print("F1 score", round(f1,5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Accuracy, Precision, Recall
# MAGIC 
# MAGIC *Precision* (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, while *recall* (also known as sensitivity) is the fraction of relevant instances that were retrieved. Both precision and recall are therefore based on relevance.
# MAGIC   
# MAGIC + Low Recall score - due to many missed Frauds  
# MAGIC + High Precision - few false positives 
# MAGIC 
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1-6Pbged7EQlCQsSzmyKYw8ApN97kcNg1" alt="drawing" width="400"/>

# COMMAND ----------

# DBTITLE 1,Use a Confusion Matrix for Assessing Categorical Models
CM = confusion_matrix(y_test, y_pred)
print(CM)

# COMMAND ----------

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

font = {'size'   : 15}
matplotlib.rc('font', **font)

ax = sns.heatmap(CM, annot=True, cmap='Blues', fmt='.8g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels');
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a Model - MLFlow
# MAGIC 
# MAGIC + *Experiments* - track all our machine-learning development runs and associated artefects
# MAGIC + *Feature-Store* - manage the data we use to train the model; retain this for model reproducibility and team collaboration
# MAGIC + *Models* - stored in the Databricks environment with namespace mapped to release cycle deployment

# COMMAND ----------

# MAGIC %md
# MAGIC install mlflow, tested with version **1.26.1**

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC If we don't set the MLflow experiment name, it is set to the notebook name.
# MAGIC   
# MAGIC Either give each MLflow training run a unique name, or let it generate the names for you.

# COMMAND ----------

import mlflow

#mlflow.set_experiment("path to experiment in remote workspace")

# COMMAND ----------

# DBTITLE 1,MLFlow model train - takes 2 minutes
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from mlflow.models.signature import infer_signature

import seaborn as sns
import numpy as np

import time

# set a maximum number of branches (speed up training time)
max_depth = 10

# assess the impact of different number of trees
estimators = [10, 20, 50]

# timestamp to append to the run-name for this set of runs
run_timestamp = int(time.time())

for n_estimators in estimators:
  with mlflow.start_run(run_name=f'cc_fraud_{n_estimators}_{run_timestamp}') as run:
    # Auto-logging to log metrics, parameters, and models without explicit log statements. silent = suppress event logs and warnings
    mlflow.sklearn.autolog(silent=True)

    # Set the hyperparameters for the model - i.e. estimators is the number of trees to use
    model_rf = RandomForestClassifier(n_estimators = n_estimators
                                     , random_state = 42
                                     , max_depth = max_depth
                                     , n_jobs = 16)
    # Train the model on training data
    model_rf.fit(X_train, y_train);

    # accuracy metrics
    y_pred = model_rf.predict(X_test).astype(int)

    accuracy = accuracy_score(y_test, y_pred)*100
    prec = precision_score(y_test, y_pred)*100
    rec = recall_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred)*100

    # set a tag so we can find this group of experiment runs
    mlflow.set_tag("project", "cc_fraud")
    
    # infer the model signature before we log it
    signature = infer_signature(X_train, model_rf.predict(X_train))

    #Log model called "cc_fraud" in the MLflow repository with parameters, and metrics
    mlflow.sklearn.log_model(model_rf
                             , "cc_fraud"
                             , input_example=X_train.loc[1:1]
                             , serialization_format = 'pickle'
                             , signature=signature
                             )
 
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', prec)
    mlflow.log_metric('recall', rec)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
   
    # example code for manually logging a confusion matrix - auto log does it for us anyway
    #CM = confusion_matrix(y_test, y_pred)
    #ax = sns.heatmap(CM, annot=True, cmap='Blues', fmt='.8g')
    #mlflow.log_figure(plt.gcf(), "test_confusion_matrix.png")
    

    # store feature importances as an artefact
    importances = model_rf.feature_importances_.round(3)
    sorted_indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns
    # setting the dimensions of the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # plot the relative importance of features
    sns.barplot(x=importances[sorted_indices]
                , y=feat_labels[sorted_indices]
                , orient='horizonal')
    for i in ax.containers:
        ax.bar_label(i,)
    mlflow.log_figure(plt.gcf(), "feature_importance.png")    

 
mlflow.end_run()  



# COMMAND ----------

# DBTITLE 1,Get best Model
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
# MAGIC ## Deploy Model Vertex AI
# MAGIC 
# MAGIC <img width="100px" src="https://techcrunch.com/wp-content/uploads/2021/05/VertexAI-512-color.png">
# MAGIC 
# MAGIC Attach `google-cloud-mlflow` from PyPi to cluster and deploy model to a Vertex AI endpoint with 3 lines of code.
# MAGIC 
# MAGIC https://www.databricks.com/blog/2022/08/12/mlops-on-databricks-with-vertex-ai-on-google-cloud.html   
# MAGIC   
# MAGIC https://github.com/Ark-kun/google_cloud_mlflow

# COMMAND ----------

# MAGIC %sh mlflow --version

# COMMAND ----------

# DBTITLE 1,Vertex AI Client for Deploying MLflow Models
import mlflow
from mlflow import deployments
from mlflow.deployments import get_deploy_client

vtx_client = mlflow.deployments.get_deploy_client("google_cloud")

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

# MAGIC %md
# MAGIC 
# MAGIC ### Deploy to a Vertex AI Endpoint
# MAGIC 
# MAGIC #### Cluster Permissions
# MAGIC The cluster must be configured with a Service Account that has privileges to deploy to the Google API endpoint.  
# MAGIC 
# MAGIC EG
# MAGIC + Vertex AI Administrator  
# MAGIC + Storage Admin  
# MAGIC 
# MAGIC (*wait about 1 minute for the permissions to take effect*)

# COMMAND ----------

# DBTITLE 1,Set Deployment Name
deployment_name = "mlflow_vertex_ccfraud"

# COMMAND ----------

# Delete deployment
deployment = vtx_client.delete_deployment(name=deployment_name)

# COMMAND ----------

# DBTITLE 1,Deploy to GCP Vertex - 5 to 10 minutes deploy time

# Create deployment
deployment = vtx_client.create_deployment(
    name=deployment_name,
    model_uri=model_uri
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src="https://drive.google.com/uc?export=view&id=1-3uLmx9S9kKMzoq1vbndYjrliAIokc1W" alt="drawing" width="1200"/>

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

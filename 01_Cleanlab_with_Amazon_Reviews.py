# Databricks notebook source
# MAGIC %md
# MAGIC # Finding and Fixing Data Issues with Cleanlab Studio
# MAGIC 
# MAGIC In this notebook, we will be looking at how to find and fix issues like mislabeled data using [Cleanlab Studio](https://cleanlab.ai/studio/).

# COMMAND ----------

!pip install git+https://github.com/anishathalye/cleanlab-studio@spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the demo dataset
# MAGIC 
# MAGIC As a demo dataset, we use a subset of the [Amazon Reviews dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), which consists of product reviews with star ratings, such as the following:
# MAGIC 
# MAGIC > I love this magazine.  Very informative and the writers are wity.  One of the few magazine that I  can read  from cover to cover.
# MAGIC >
# MAGIC > (5 stars)
# MAGIC 
# MAGIC Some of the reviews have incorrect ratings, due to user error:
# MAGIC 
# MAGIC > One of my favorite magazines I've purchased for the last thirty five years.
# MAGIC >
# MAGIC > (3 stars)
# MAGIC 
# MAGIC This user likely meant to give the magazine a 5-star rating, but mis-clicked. Let's see how we can find such erroneously-labeled datapoints and correct their labels.
# MAGIC 
# MAGIC We load the dataset into a [PySpark DataFrame](https://docs.databricks.com/getting-started/dataframes-python.html). In this tutorial, the data is read from an external URL. When you apply this to your own data, it'll be loaded directly from your data in Databricks, e.g., using `spark.sql(...)`.

# COMMAND ----------

!wget -N https://s.cleanlab.ai/amazon-text-demo.csv -P /dbfs/tmp/solacc/confidence_learning/amazon-text-demo.csv

# COMMAND ----------

from pyspark import SparkFiles
spark.sparkContext.addFile('https://s.cleanlab.ai/amazon-text-demo.csv')
amazon_reviews = spark.read.csv("/tmp/solacc/confidence_learning/amazon-text-demo.csv", header=True, inferSchema=True)
display(amazon_reviews)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Cleanlab Studio
# MAGIC 
# MAGIC 1. If you don't have an account already, [sign up for an account](https://app.cleanlab.ai/)
# MAGIC 2. Get your [API key](https://app.cleanlab.ai/account?tab=General). Never leave a credential in the notebook in plain text 
# MAGIC 3. Run the cells below to install the Cleanlab Studio connector

# COMMAND ----------

CLEANLAB_STUDIO_API_KEY = dbutils.secrets.get("solution-accelerator-cicd", "cleanlab_api") 

# COMMAND ----------

from cleanlab_studio import Studio, Schema
studio = Studio(CLEANLAB_STUDIO_API_KEY)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upload data to Cleanlab Studio
# MAGIC 
# MAGIC Follow two steps to upload data to Cleanlab Studio so it can be analyzed:
# MAGIC 
# MAGIC 1. Supply a schema for the dataset (describing the type of each column); this can be automatically inferred using `Schema.infer`, as shown below
# MAGIC 2. Call `upload_dataset`

# COMMAND ----------

schema = Schema.infer(amazon_reviews, name='Amazon Reviews', modality='text')
schema.to_dict() # you can review the schema to make sure it looks reasonable

# COMMAND ----------

# now, upload the dataset
dataset_id = studio.upload_text_dataset(amazon_reviews, schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Project
# MAGIC 
# MAGIC To analyze the data, use the Cleanlab Studio web UI to create a project, configuring it according to your ML task and other preferences. For this demo, you can create a project based on your uploaded Amazon Reviews dataset, selecting:
# MAGIC 
# MAGIC - ML task: text classification
# MAGIC - Type of classification: multi-class
# MAGIC - Text column: review_text
# MAGIC - Label column: stars
# MAGIC 
# MAGIC Select fast mode or regular mode depending on the speed/quality tradeoff you desire.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Make Corrections
# MAGIC 
# MAGIC Cleanlab Studio not only finds data points with potential issues, but it also makes suggestions for how to address the issues (e.g., changing the label of a data point). Deciding how to make use of the analysis results is up to you. For example, you could discard all potentially erroneous data points, or you could review the data points most likely to have issues and make corrections. This human-in-the-loop data correction usually gives the best results.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Results
# MAGIC 
# MAGIC Once you're done cleaning data, you've created a "Cleanset". You can export the results by clicking on the "Export Cleanset" button, getting the project ID, and entering it below. You can either download raw analysis results, or you can apply the corrections made in the web interface to your original dataset.

# COMMAND ----------

CLEANSET_ID = '9843100d9e324caf9fa17a5c15d8130a'

# COMMAND ----------

reviews_fixed = studio.apply_corrections(CLEANSET_ID, amazon_reviews)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's inspect some specific errors:

# COMMAND ----------

review_id = 'review16896'
display(amazon_reviews.filter(amazon_reviews.index == review_id))
display(reviews_fixed.filter(reviews_fixed.index == review_id))

# COMMAND ----------

# MAGIC %md
# MAGIC We can also download raw analysis results.

# COMMAND ----------

cl_cols = studio.download_cleanlab_columns('9e3d31d7c6384726b1d772af5f136c05')
cl_cols

# COMMAND ----------



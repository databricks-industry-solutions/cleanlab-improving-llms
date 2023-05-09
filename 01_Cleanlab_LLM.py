# Databricks notebook source
# MAGIC %md
# MAGIC # Better Large Language Models (LLMs) With Better Data
# MAGIC
# MAGIC This notebook demonstrates how to fine-tune large language models (LLMs) from your Databricks Notebooks, and how Databricks can integrate with [data-centric AI](https://dcai.csail.mit.edu/) tools like [Cleanlab Studio](https://app.cleanlab.ai/) that can improve the performance of your LLMs by improving the data quality.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This notebook focuses on _fine-tuning LLMs_. LLMs acquire powerful generative and discriminative capabilities after being pre-trained on a large corpus of text (usually scraped from the internet), but producing reliable outputs for a particular business use case often requires additional training on a labeled data set from the application domain. This domain-specific training is known as _fine-tuning_ the LLM.
# MAGIC
# MAGIC Labeled data powers AI/ML in the enterprise, but real-world datasets have been found to contain between 7-50% annotation errors. Imperfectly-labeled text data hampers the training (and evaluation of) ML models across tasks like intent recognition, entity recognition, and sequence generation. Although pretrained LLMs are equipped with a lot of world knowledge, their performance is adversely affected by noisy training data (as [noted by OpenAI](https://openai.com/research/dall-e-2-pre-training-mitigations)).  This notebook illustrate data-centric techniques to mitigate the effect of low-quality data without changing any code related to model architecture, hyperparameters, or training. These data quality improvement techniques should thus remain applicable even for future advanced LLMs like GPT-10.
# MAGIC
# MAGIC This notebook applies LLMs to a politeness classification task, beginning by fine-tuning OpenAI's Davinci model on the baseline dataset. The model achieves moderate performance on this baseline, but by automatically finding and fixing errors in the data using the Databricks connector for [Cleanlab Studio](https://app.cleanlab.ai/), we can achieve significantly better performance _using the same LLM model and fine-tuning process_, just by improving the data (and spending minimal human time on manually reviewing data). We see a performance of 37% when using Cleanlab Studio to improve the dataset:
# MAGIC
# MAGIC # TODO: update chart and remove middle bar
# MAGIC ![](https://i.imgur.com/ZQ9WijM.png)
# MAGIC
# MAGIC See the accompanying blog post for additional context on LLMs and fine-tuning, why data quality matters for LLMs and ML tasks in general, and how data-centric AI techniques and tools can help you easily improve ML model robustness and performance by systematically improving data quality.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC We will use Cleanlab Studio to enhance data quality in this notebook. If you don't have a Cleanlab Studio account already, [sign up for an account here](https://app.cleanlab.ai/). It may take up to one day to get access.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install and configure dependencies
# MAGIC
# MAGIC This notebook uses the fine-tuning APIs [offered by OpenAI](https://platform.openai.com/docs/guides/fine-tuning).

# COMMAND ----------

!pip install openai

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure OpenAI API key
# MAGIC
# MAGIC Note that invoking the OpenAI API will use credits or bill you. The estimated cost to run this notebook is $15 with the Davinci model, which is the most powerful but also the most expensive. You can also scale down to the Curie or Ada model to reduce the cost, by replacing "davinci" with "curie" or "ada" in the invocations of the fine-tuning API. Fine-tuning on the Ada model costs about $1 per run with the given dataset.
# MAGIC
# MAGIC Put your OpenAI API key in the cell below. You can find your API key at https://platform.openai.com/account/api-keys. Here we have saved the key in a secret scope - see the `RUNME` notebook in this repository for helper scripts to set up the secret scope.

# COMMAND ----------

import openai
import os

# we set the environment variable because it is used by the OpenAI command-line tool
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("solution-accelerator-cicd","openai_api")
# we also set the .api_key property below for the Python API
openai.api_key = os.environ['OPENAI_API_KEY']
# set openai model name
openai_model = 'curie'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download and prepare data
# MAGIC
# MAGIC Here we consider a 3-class variant of the Stanford Politeness Dataset, which has text phrases labeled as: impolite, neutral, or polite. Annotated by human raters, some of these labels are naturally low-quality. 
# MAGIC
# MAGIC The training dataset has 1916 examples each labeled by a single human annotator, and thus some may be unreliable.
# MAGIC
# MAGIC The test dataset has 480 examples each labeled by 5 annotators, and we use their consensus label as a high-quality approximation of the true politeness (measuring test accuracy against these consensus labels). To ensure a fair comparison, this test dataset remains fixed throughout the experiments in this notebook (all data cleaning is done only for the training set).
# MAGIC
# MAGIC To prepare the data, we download raw data into [DBFS](https://docs.databricks.com/dbfs/index.html), load it into PySpark DataFrames, and do some processing to prepare the dataset for the downstream task.

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC rm -rf /tmp/stanford-politeness
# MAGIC mkdir -p /tmp/stanford-politeness
# MAGIC cd /tmp/stanford-politeness
# MAGIC curl --silent -L https://s.cleanlab.ai/stanford-politeness/fine-tuning/train.csv -o train.csv
# MAGIC curl --silent -L https://s.cleanlab.ai/stanford-politeness/fine-tuning/test.csv -o test.csv
# MAGIC
# MAGIC # move the dataset to our main bucket
# MAGIC rm -rf /dbfs/solacc/product/llm/stanford-politeness/raw
# MAGIC mkdir -p /dbfs/solacc/product/llm/stanford-politeness/raw
# MAGIC cp train.csv test.csv /dbfs/solacc/product/llm/stanford-politeness/raw

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can use the `%fs` command to see that our raw data is indeed saved in DBFS.

# COMMAND ----------

# MAGIC %fs ls /solacc/product/llm/stanford-politeness/raw

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we load the raw data into a PySpark DataFrame to enable further processing.

# COMMAND ----------

data_path = '/solacc/product/llm/stanford-politeness'
raw_path = f'{data_path}/raw'
politeness_train_raw = spark.read.options(header='true', inferSchema='true', escape='"', multiLine=True).csv(f'{raw_path}/train.csv')
politeness_test_raw = spark.read.options(header='true', inferSchema='true', escape='"', multiLine=True).csv(f'{raw_path}/test.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC This dataset is missing an index column, but downstream processing that we plan to do requires a unique ID per row. Here, we add monotonically-increasing integer IDs to the rows.

# COMMAND ----------

from pyspark.sql import functions as F

def with_id_column(df):
    df = df.select(F.monotonically_increasing_id().alias("id"), "*")
    return df

politeness_train = with_id_column(politeness_train_raw)
politeness_test = with_id_column(politeness_test_raw)

# COMMAND ----------

# MAGIC %md
# MAGIC We can inspect this prepared data, looking at some specific rows to highlight data errors that are present. For example, the data point with ID `1426` is erroneously labeled "impolite".

# COMMAND ----------

display(politeness_train.where((politeness_train.id == 1426) | (politeness_train.id == 299) | (politeness_train.id == 134)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Formatting data for fine-tuning
# MAGIC
# MAGIC We are using the OpenAI APIs for fine-tuning, which require data in a specific format (JSONL). We also need to do some pre-processing of the label column, adding whitespace before the completion, as the API recommends.
# MAGIC
# MAGIC We save the prepared results into DBFS, so that the result files can be used by the OpenAI API.

# COMMAND ----------

def prepare_data(df, path):
    '''
    Write a dataframe into a single JSONL file located at path, in a format appropriate for fine tuning.

    This makes a small tweak to the data, namely, adding whitespace before the completion.

    By default, spark writes the dataset into multiple files for efficiency, but we need a single file to pass to the OpenAI command-line tool.
    '''
    # add whitespace to the completion, as OpenAI requires
    df = df.withColumn('completion', F.format_string(' %s', 'completion'))
    # we don't need the index column here
    df = df.drop('id')
    temp_dir = f'{path}_tmp'
    # write using a single partition, so we have a single JSONL file
    df.coalesce(1).write.mode('overwrite').json(temp_dir)
    # Spark saves the JSON file in a directory, along with some other files we don't need anymore
    all_files = dbutils.fs.ls(temp_dir)
    for f in all_files:
        if f.path.endswith('.json'):
            dbutils.fs.mv(f.path, path)
    # remove all the files we don't need
    dbutils.fs.rm(temp_dir, recurse=True)

# COMMAND ----------

prepare_data(politeness_train, f'{data_path}/processed/train.jsonl')
prepare_data(politeness_test, f'{data_path}/processed/test.jsonl')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-Tune and Evaluate OpenAI Model without Cleanlab Studio (accuracy 63%)
# MAGIC
# MAGIC We use the [OpenAI fine-tuning API](https://platform.openai.com/docs/guides/fine-tuning) to first establish a baseline by:
# MAGIC
# MAGIC - Fine-tuning the OpenAI model on our (original, with some errors) training set
# MAGIC - Evaluating the model on our test set

# COMMAND ----------

train_file = openai.File.create(file=open(f'/dbfs/{data_path}/processed/train.jsonl', 'rb'), purpose='fine-tune')
test_file = openai.File.create(file=open(f'/dbfs/{data_path}/processed/test.jsonl', 'rb'), purpose='fine-tune')

# COMMAND ----------

response = openai.FineTune.create(
    training_file=train_file.id,
    validation_file=test_file.id,
    compute_classification_metrics=True,
    classification_n_classes=3,
    model=openai_model,
    suffix='baseline'
)

# COMMAND ----------

# MAGIC %md
# MAGIC You can follow the progress of fine-tuning with the following command. Once it's done, it'll print "Job complete!". You might need to re-run the cell if it times out. Training time varies based on queue length and other factors; **it can take up to 1 hour to fine-tune the LLM**. The block below would check the status of the finetune and block execution until the finetune is complete. The block is based on this [openai-cookbook example](https://github.com/openai/openai-cookbook/blob/594fc6c952425810e9ea5bd1a275c8ca5f32e8f9/examples/azure/finetuning.ipynb#L278).

# COMMAND ----------

import time
job_id = response.id

status = openai.FineTune.retrieve(id=job_id)["status"]
if status not in ["succeeded", "failed"]:
    print(f'Job not in terminal status: {status}. Waiting.')
    while status not in ["succeeded", "failed"]:
        time.sleep(60)
        status = openai.FineTune.retrieve(id=job_id)["status"]
        print(f'Status: {status}')
else:
    print(f'Finetune job {job_id} finished with status: {status}')


# COMMAND ----------

# MAGIC %md
# MAGIC Once the job completes, we see the test accuracy achieved when fine-tuning this LLM on the original training dataset.

# COMMAND ----------

import pandas as pd

!openai api fine_tunes.results -i {response.id} > baseline.csv

base_df = pd.read_csv('baseline.csv')
baseline_acc = base_df.iloc[-1]['classification/accuracy']
print(f"Fine-tuning Accuracy: {baseline_acc:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC Our baseline Davinci LLM achieves a test accuracy of 63% when fine-tuned on the original training data. Even a state-of-the-art LLM like the Davinci model produces lackluster results for this classification task; is it because of low data quality? 
# MAGIC
# MAGIC TODO: talk about how individual run performance may vary. History: 
# MAGIC
# MAGIC davinci 'ft-WXKwsuxRpbDXWc3MRhWDoAuE': 65%
# MAGIC
# MAGIC ada 60%
# MAGIC
# MAGIC Curie 64.2%

# COMMAND ----------

# MAGIC %md
# MAGIC ## Improve the data using Cleanlab Studio and re-train the LLM (accuracy 73%)
# MAGIC
# MAGIC Next, we use the [Databricks connector](https://github.com/cleanlab/cleanlab-studio) for [Cleanlab Studio](https://app.cleanlab.ai/) to automatically improve the data quality, and then re-train our LLM.

# COMMAND ----------

!pip install cleanlab-studio
import cleanlab_studio

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set up Cleanlab Studio
# MAGIC
# MAGIC 1. If you don't have an account already, [sign up for an account](https://app.cleanlab.ai/). It may take up to one day to get access.
# MAGIC 2. Get your [API key](https://app.cleanlab.ai/account?tab=General) and enter it below

# COMMAND ----------

CLEANLAB_STUDIO_API_KEY = dbutils.secrets.get("solution-accelerator-cicd","cleanlab_api")
studio = cleanlab_studio.Studio(CLEANLAB_STUDIO_API_KEY)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload dataset to Cleanlab Studio
# MAGIC Next, we can directly upload a Spark DataFrame to Cleanlab Studio by passing it to `studio.upload_dataset()`.

# COMMAND ----------

dataset_id = studio.upload_dataset(politeness_train, dataset_name='Stanford Politeness')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Project
# MAGIC
# MAGIC To analyze the data, use the [Cleanlab Studio web UI](https://app.cleanlab.ai/) to create a project, configuring it according to the ML task. For this demo, you should select:
# MAGIC
# MAGIC - ML task: text classification
# MAGIC - Type of classification: multi-class
# MAGIC - Text column: prompt (will be auto-detected)
# MAGIC - Label column: completion (will be auto-detected)
# MAGIC
# MAGIC Select fast mode or regular mode depending on the speed/quality tradeoff you desire.
# MAGIC
# MAGIC ![](https://i.imgur.com/WjFTQms.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Make corrections
# MAGIC
# MAGIC Cleanlab Studio not only finds data points with potential issues, but it also makes suggestions for how to address the issues (e.g., changing the label of a data point). Deciding how to make use of the analysis results is up to you. For example, you could discard all potentially erroneous data points, or you could review the data points most likely to have issues and make corrections. This human-in-the-loop data correction usually gives the best results.
# MAGIC
# MAGIC If you want to save time, you could briefly review some flagged issues, and then auto-fix the top issues.
# MAGIC
# MAGIC ![](https://i.imgur.com/EUNueDg.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export your improved dataset back to Databricks
# MAGIC
# MAGIC Once you're done correcting issues found in your dataset with Cleanlab Studio, export the improved dataset by clicking on the "Export Cleanset" button within your Cleanlab Studio project. Next select the "Export using API" tab and copy the "cleanset ID" and paste it in the cell below.

# COMMAND ----------

# cleanset_id = 'PASTE CLEANSET ID HERE'
# politeness_train_fixed = studio.apply_corrections(cleanset_id, politeness_train)
# display(politeness_train_fixed)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fine-tune the LLM on your improved dataset and evaluate the results
# MAGIC
# MAGIC Let's see how Cleanlab Studio improves the performance of the LLM. We follow the same process as earlier, except we use the `politeness_train_fixed` DataFrame as our training data.
# MAGIC
# MAGIC When we ran the experiment below, we used Cleanlab Studio's web interface to review the data issues that it flagged. Machine-augmented human-in-the-loop data improvement often gives the best results. If you want to use the dataset that we exported from Cleanlab Studio, uncomment the line below.

# COMMAND ----------

# by default, use your dataset that you improved, downloaded as politeness_train_fixed above
#
# but for reproducibility, if you want to use the dataset that we exported from Cleanlab Studio,
# set the flag below to 'True'
use_provided_training_set_improved_using_cleanlab_studio = True
if use_provided_training_set_improved_using_cleanlab_studio:
    politeness_train_fixed = pd.read_csv('https://s.cleanlab.ai/stanford-politeness/fine-tuning/train_fixed.csv')
    politeness_train_fixed = with_id_column(spark.createDataFrame(politeness_train_fixed))

# COMMAND ----------

prepare_data(politeness_train_fixed, f'{data_path}/processed/train_fixed.jsonl')

# COMMAND ----------

train_file_fixed = openai.File.create(file=open(f'/dbfs/{data_path}/processed/train_fixed.jsonl', 'rb'), purpose='fine-tune')

# COMMAND ----------

response_fixed = openai.FineTune.create(
    training_file=train_file_fixed.id,
    validation_file=test_file.id,
    compute_classification_metrics=True,
    classification_n_classes=3,
    model=openai_model,
    suffix='fixed'
)

# COMMAND ----------

# MAGIC %md
# MAGIC You can follow the progress of fine-tuning with the following command. Once it's done, it'll print "Job complete!". You might need to re-run the cell if it times out. Training time varies based on queue length and other factors; it can take up to 1 hour to fine-tune the LLM.

# COMMAND ----------

import time
job_id = response_fixed.id

status = openai.FineTune.retrieve(id=job_id)["status"]
if status not in ["succeeded", "failed"]:
    print(f'Job not in terminal status: {status}. Waiting.')
    while status not in ["succeeded", "failed"]:
        time.sleep(60)
        status = openai.FineTune.retrieve(id=job_id)["status"]
        print(f'Status: {status}')
else:
    print(f'Finetune job {job_id} finished with status: {status}')


# COMMAND ----------

# MAGIC %md
# MAGIC Once the job completes, we see the test accuracy achieved when fine-tuning this LLM on the improved dataset. If you simply auto-fixed some of the labels (spending zero human time on data improvement), you'll still see improvement; if you reviewed some of Cleanlab Studio's suggestions following a human-in-the-loop data cleaning process, you'll see large improvements here.

# COMMAND ----------

# change to True to load dataset from our 


# COMMAND ----------

!openai api fine_tunes.results -i {response_fixed.id} > fixed.csv

fixed_df = pd.read_csv('fixed.csv')
fixed_acc = fixed_df.iloc[-1]['classification/accuracy']
print(f"Fine-tuning Accuracy: {fixed_acc:.1%}")

# COMMAND ----------

# MAGIC %md TODO: 
# MAGIC
# MAGIC Talk about performance for different models
# MAGIC
# MAGIC davinci 77.9%
# MAGIC
# MAGIC ada 74.8%
# MAGIC
# MAGIC Curie 75.8%

# COMMAND ----------

# MAGIC %md
# MAGIC # Takeaways: what should I do differently going forward?
# MAGIC
# MAGIC Data-centric AI is a powerful paradigm for handling noisy data via AI/automated techniques rather than the tedious manual effort data scientists often dread. Tools like [Cleanlab Studio](https://app.cleanlab.ai/) help you efficiently find and fix data and label issues that can be used to improve any ML model (not just LLMs) for most types of data (not just text, but also images, audio, tabular data, etc). Cleanlab Studio improves your data without requiring that you write code or have any ML expertise.
# MAGIC
# MAGIC These sorts of tools will still remain applicable with future advances in ML models like GPT-10, and will only become better at identifying issues when used with more accurate models!  Practice data-centric AI to systematically engineer better data via AI/automation. This frees you to capitalize on your unique domain knowledge rather than fixing general data issues like label errors.

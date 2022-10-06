# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Train reliable ML models to understand human emotion with [Cleanlab](https://docs.cleanlab.ai/)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Machine learning (ML) models that understand human emotion can automate and improve:
# MAGIC 
# MAGIC .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; .-.-.-.-.-.-.-.-.-.-.-.-.
# MAGIC <br> 
# MAGIC | &nbsp;&nbsp; medical diagnoses &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
# MAGIC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp; emotion AI &nbsp;&nbsp; | <br>
# MAGIC | &nbsp;&nbsp; sales &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.-.-.-.-.-.-.-.-.-.-.-.'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;&nbsp; interviewing &nbsp;|
# MAGIC <br>| &nbsp;&nbsp; therapy&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp; teaching &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
# MAGIC <br>
# MAGIC | &nbsp;&nbsp; chatbots &nbsp;&nbsp;\`-'-'-'-'-'-'-'-'-'-'-'--.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp; affective AI &nbsp;&nbsp;\`-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'--.<br> 
# MAGIC | &nbsp;&nbsp; mental health support &nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;&nbsp; customer experience and feedback &nbsp;&nbsp;| <br>
# MAGIC \`-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-' &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\`-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'
# MAGIC 
# MAGIC **The problem is...** AI systems for real-world, human-centric data tend to yield lackluster performance, not becuase of the algorithm/models, but because real-world, human-centric data tends to be low quality due to [label errors](https://labelerrors.com/) and [other issues](https://docs.cleanlab.ai/master/cleanlab/outlier.html). 

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC # In this tutorial...
# MAGIC 
# MAGIC You'll learn how to use [cleanlab](https://github.com/cleanlab/cleanlab) to: 
# MAGIC * automatically find label errors in the Google Emotions dataset
# MAGIC * train your ML model on clean data using [CleanLearning()](https://docs.cleanlab.ai/v2.0.0/cleanlab/classification.html#cleanlab.classification.CleanLearning)
# MAGIC * improve the reliability and accuracy of an emotion detection AI classifier!

# COMMAND ----------

# MAGIC %pip install cleanlab autogluon.tabular[lightgbm,xgboost]

# COMMAND ----------

import os
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from cleanlab.filter import find_label_issues
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
import cleanlab
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option("display.max_colwidth", None)
os.chdir("/databricks/driver")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC # The dataset
# MAGIC Download the Google Emotions dataset here: https://github.com/cleanlab/datasets/tree/main/go_emotions_subset
# MAGIC * comments were extracted from Reddit with human annotations for 28 emotion categories. Here we select 4 of them.
# MAGIC * we augmented the dataset by adding the sentiment for every comment (thanks [HuggingFace](https://huggingface.co/docs/transformers/v4.22.1/en/main_classes/pipelines#transformers.pipeline.example) 🤗)

# COMMAND ----------

# MAGIC %run /util/data_etl

# COMMAND ----------

import os
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from cleanlab.filter import find_label_issues
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
import cleanlab
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option("display.max_colwidth", None)
os.chdir("/databricks/driver")

# COMMAND ----------

# These are the four emotions. 
target_emotions = ['annoyance', 'curiosity', 'disapproval', 'neutral']
train_path = "/dbfs/tmp/emotionai/train_data_balanced.csv"
test_path = "/dbfs/tmp/emotionai/test_data_balanced.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC Both datasets contain 4 columns:
# MAGIC 
# MAGIC * `comment_text`: the text of the comment, with masked tokens for names, private info, etc.
# MAGIC * `sentiment`: float in the range [0,1] calculated by HuggingFace sentiment analysis pipeline. Smaller number means more negative.
# MAGIC * `label`: numerical label of the text, in range [0, 4]
# MAGIC * `emotion`: text representation of the label 
# MAGIC 
# MAGIC 
# MAGIC # 

# COMMAND ----------

# DBTITLE 1,Let's look at some hand-picked examples from the dataset:
train_data = pd.read_csv(train_path).drop("Unnamed: 0", axis=1)
test_data = pd.read_csv(test_path).drop("Unnamed: 0", axis=1)
# Hand-picked examples of label errors in the data (found via Cleanlab)
display(train_data.iloc[[277,7673, 8121, 8911, 9088]])

# COMMAND ----------

# MAGIC %md
# MAGIC # Are the emotion labels different than what you expected?
# MAGIC It turns out real-world datasets are often [full of errors](https://labelerrors.com/). As we can see, the Google Emotions dataset also contains many label errors.
# MAGIC 
# MAGIC # Steps to train a reliable emotion understanding model on error-prone data:
# MAGIC 1. Try training a model on the original dataset
# MAGIC 2. Use [cleanlab](https://github.com/cleanlab/cleanlab) to automatically find the label issues
# MAGIC 3. Train a robust model on data we are confident is labeled correctly
# MAGIC 4. Is there improvement for class-accuracies and overall accuracy?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train an ML model on multi-modal (text and tabular) data
# MAGIC We leverage the [`autogluon.tabular`](https://auto.gluon.ai/stable/api/autogluon.predictor.html#autogluon.tabular.TabularPredictor) predictor to allow us to train a fuse-model on both text and tabular (the sentiment float values) simultaneously. The tabular predictor combines both dataset sources into a single embedded representation for end-to-end training.

# COMMAND ----------

# The data includes this column for easy emotion recognition. Don't need for training.
X_train = train_data.drop(["emotion"], axis =1)
X_test = test_data.drop(["emotion"], axis =1)
# Initiate AutoGluon tabular predictor by specifying the label column.
predictor = TabularPredictor(label="label", verbosity=0)
# We tried a few models here and these seemed to work pretty well on this task.
hyperparameters = {
    "XGB": {},
    "GBM": [{"extra_trees": True, "ag_args": {"name_suffix": "XT"}}, {}],
}

# COMMAND ----------

# MAGIC %md
# MAGIC # Train our ML models on the original **data** without cleanlab...

# COMMAND ----------

predictor.fit(train_data=X_train, tuning_data=X_test, hyperparameters=hyperparameters)
predictor.leaderboard()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute overall accuracy and per-class accuracies

# COMMAND ----------

# Get labels.
y_true = X_test["label"]
# Drop label col for prediction.
test_data_nolab = X_test.drop("label", axis=1)
# Make prediction and evaluate performance
y_pred = predictor.predict(test_data_nolab)
perf = predictor.evaluate_predictions(
    y_true=y_true, y_pred=y_pred, auxiliary_metrics=True
)
base_acc = perf["accuracy"]
matrix = confusion_matrix(y_true, y_pred)
# Per-class accuracies are important!
class_acc = matrix.diagonal() / matrix.sum(axis=1)
class_acc_str = [f"{acc:0.1%}" for acc in class_acc]

print(f"Accuracy without Cleanlab: {base_acc:.1%}")
print(class_acc_str)

# COMMAND ----------

# MAGIC %md
# MAGIC ## How did our ML model do? (without Cleanlab)
# MAGIC 
# MAGIC * **Overall Test Accuracy** (without Cleanlab):
# MAGIC   * 67.5%
# MAGIC * **Class Test Accuracies** (without Cleanlab): 
# MAGIC   * annoyance: 84.5%
# MAGIC   * curiosity: 61.9%
# MAGIC   * disapproval: 61.8%
# MAGIC   * neutral: 59.9%

# COMMAND ----------

# MAGIC %md
# MAGIC # Train a more reliable model using cleanlab to provide clean data for training!
# MAGIC 
# MAGIC Let's use cleanlab to improve the quality of our data.
# MAGIC 
# MAGIC First, we need out-of-sample predicted probabilities for our training set.
# MAGIC 
# MAGIC * The cell block below takes ~2min.

# COMMAND ----------

del predictor  # Helps free up memory.

def get_pred_probs():
    """Uses cross-validation to obtain out-of-sample predicted probabilities
    for our entire dataset"""

    hyperparameters = {
        "XGB": {},
        "GBM": [{"extra_trees": True, "ag_args": {"name_suffix": "XT"}}, {}],
    }
    num_examples, num_classes = X_train.shape[0], len(X_train.label.value_counts())
    skf = StratifiedKFold()
    skf_splits = [
        [train_index, test_index]
        for train_index, test_index in skf.split(X=X_train, y=X_train["label"])
    ]
    pred_probs = pd.DataFrame(
        np.zeros((num_examples, num_classes)),
        columns=[i for i in range(len(target_emotions))],
    )

    # Iterate through cross-validation folds
    for split_num, split in enumerate(skf_splits):
        train_index, val_index = split
        train_data_subset = X_train.iloc[train_index]
        validation_data = X_train.iloc[val_index]
        predictor = TabularPredictor(label="label", verbosity=0)
        predictor.fit(
            train_data_subset,
            tuning_data=validation_data,
            hyperparameters=hyperparameters,
        )
        pred_probs_fold = predictor.predict_proba(validation_data, as_pandas=True)
        pred_probs.iloc[val_index] = pred_probs_fold
        del pred_probs_fold
        del predictor
    return pred_probs.values

# Out-of-sample predicted probabilities
pred_probs = get_pred_probs()

# COMMAND ----------

# MAGIC %md
# MAGIC # Option 1 - Find label issues with [`CleanLearning`](https://docs.cleanlab.ai/stable/cleanlab/classification.html#cleanlab.classification.CleanLearning)
# MAGIC 
# MAGIC Learn more: https://docs.cleanlab.ai/stable/index.html#find-label-errors-in-your-data
# MAGIC 
# MAGIC 
# MAGIC Cleanlab found <font color='red'>1,896</font> label problems using `filter_by="confident_learning"`. Let's take a look.

# COMMAND ----------

cl = CleanLearning(find_label_issues_kwargs={"filter_by": "both", "frac_noise":0.8})
# Label errors found automatically in this one line of code.
df = cl.find_label_issues(labels=X_train.label.values, pred_probs=pred_probs)

errs = sum(df.is_label_issue)
print(f"Cleanlab found {errs} ({errs / len(train_data):0.1%}) potential label errors.")
# Replace numerical labels with emotion names and drop unused columns.
df["given_emotion"] = df.given_label.apply(lambda x: target_emotions[x])
df["suggested_emotion"] = df.predicted_label.apply(lambda x: target_emotions[x])
df = df.drop(["given_label", "predicted_label", "is_label_issue"], axis=1)
df.insert(0, "comment_text", train_data.loc[df.index.values].comment_text)
df.sort_values(by=["label_quality"]).head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Option 2 - Find label issues with [`filter.find_label_issues()`](https://docs.cleanlab.ai/stable/cleanlab/filter.html#cleanlab.filter.find_label_issues)
# MAGIC 
# MAGIC 
# MAGIC Learn more: https://docs.cleanlab.ai/stable/index.html#find-label-errors-in-your-data.
# MAGIC 
# MAGIC This is the lower-level method used by `CleanLearning`. Depending on your workflow, it may be easier to work with this direclty.

# COMMAND ----------

ordered_label_issues = find_label_issues(
    labels=train_data.label.values,
    pred_probs=pred_probs,
    filter_by="both",
    frac_noise = 0.8,
    return_indices_ranked_by="self_confidence",
)
print(f"5 comments with lowest quality labels:")
idx = ordered_label_issues[:5]
labels = [target_emotions[i] for i in train_data.iloc[idx].label.values]
texts = train_data.iloc[idx].comment_text.values
pd.DataFrame({"text": texts, "label": labels}, [idx])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a reliable ML model by providing clean data during training using cleanlab

# COMMAND ----------

# Remove error-prone labels from training set.
cleanset_idx = ordered_label_issues
clean_train_data = X_train.drop(cleanset_idx)

# Train the models with cleaned data (we are actually training with LESS data!)
predictor = TabularPredictor(label="label", verbosity=0)
predictor.fit(clean_train_data, tuning_data=X_test, hyperparameters=hyperparameters)

# COMMAND ----------

# Test model
y_true = X_test["label"]
test_data = X_test.drop("label", axis=1)
y_pred = predictor.predict(test_data)
perf = predictor.evaluate_predictions(
    y_true=y_true, y_pred=y_pred, auxiliary_metrics=True
)
clean_acc = perf["accuracy"]
matrix = confusion_matrix(y_true, y_pred)
clean_class_acc = matrix.diagonal() / matrix.sum(axis=1)
clean_class_acc_str = [f"{acc:0.1%}" for acc in clean_class_acc]

num_errs = len(ordered_label_issues)
print(f"Cleanlab removed {num_errs} datapoints. This is {num_errs / len(X_train):.1%} of the original dataset.")
print("Base Accuracy:", "{:.1%}".format(base_acc), "\nClean Accuracy:", "{:.1%}".format(clean_acc))
print("Cleanlab Improvement:", "{:.1%}".format(clean_acc - base_acc), "\n")
print("Base class acc:", class_acc_str, "\nClean class acc:", clean_class_acc_str)
print("Class Improvement:", [f"{z:.1%}" for z in clean_class_acc - class_acc])


# COMMAND ----------

# MAGIC %md
# MAGIC ## How did our ML model do? (**with** Cleanlab)
# MAGIC 
# MAGIC * **Overall Test Accuracy** (**with** Cleanlab):
# MAGIC   * 69.4% <font color='gree'>+1.7%</font>
# MAGIC 
# MAGIC * Class Test Accuracies (**with** Cleanlab):
# MAGIC   * annoyance: 85.9% <font color='gree'>+1.3%</font>
# MAGIC   * curiosity: 62.3% <font color='gree'>+0.4%</font>
# MAGIC   * disapproval: 63.8% <font color='gree'>+2.0%</font>
# MAGIC   * neutral: 63.0% <font color='gree'>+3.0%</font>
# MAGIC 
# MAGIC Overall increase of <font color='gree'>~2%</font> -- with multiple <font color='gree'>1%-3%</font> class improvements!

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC # [Cleanlab Studio](https://cleanlab.ai/studio/) -- automatic data correction with zero lines of code.
# MAGIC 
# MAGIC ## In this tutorial, we saw how to use [cleanlab](https://github.com/cleanlab/cleanlab) to **find** label issues -- to **fix/correct** your data and train a more reliable model on your entire dataset, you need an interface to interact with the data.
# MAGIC 
# MAGIC * **Cleanlab Studio** finds **and** fixes label issues automatically in a (very cool) no-code platform.
# MAGIC * Export your corrected dataset with a click to train better ML models on better data.
# MAGIC 
# MAGIC Try Cleanlab Studio at https://cleanlab.ai/studio/.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC # Resources
# MAGIC 
# MAGIC * [Documentation](https://docs.cleanlab.ai/) | [Blogs](https://cleanlab.ai/blog/) | [Research Publications](https://cleanlab.ai/research/)
# MAGIC * Step-by-step tutorials to find issues in your data and train robust ML models:
# MAGIC   * [Image](https://docs.cleanlab.ai/stable/tutorials/image.html) | [Text](https://docs.cleanlab.ai/stable/tutorials/text.html) | [Audio](https://docs.cleanlab.ai/stable/tutorials/audio.html)
# MAGIC * Ways to try out Cleanlab:
# MAGIC   * Open-source: [GitHub](https://github.com/cleanlab/cleanlab)
# MAGIC   * No-code, automatic platform (easy mode): [Cleanlab Studio](https://cleanlab.ai/studio/)
# MAGIC   * Learn how Cleanlab works: [Cleanlab Vizzy](https://playground.cleanlab.ai/) 
# MAGIC 
# MAGIC Join our (amazing) community of scientists and engineers: [Cleanlab Slack Community](https://cleanlab.ai/slack)

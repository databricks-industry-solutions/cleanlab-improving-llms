# Databricks notebook source
# MAGIC %pip install transformers==4.22.1 tensorflow==2.10.0 tensorflow-datasets==4.6.0 Jinja2==3.1.2

# COMMAND ----------

from transformers import pipeline
import tensorflow_datasets as tfds
from numpy.random import RandomState
import numpy as np
import pandas as pd
import random
tfds.disable_progress_bar()

# COMMAND ----------

# Get GoEmotions from tfds.
raw_train_ds = tfds.load('goemotions', split='train', data_dir = "/databricks/driver/goemotions")
raw_test_ds = tfds.load('goemotions', split='test', data_dir = "/databricks/driver/goemotions")
raw_val_ds = tfds.load('goemotions', split='validation', data_dir = "/databricks/driver/goemotions")

# COMMAND ----------

# Convert to Dataframe.
df_train = pd.DataFrame(tfds.as_dataframe(raw_train_ds))
df_test = pd.DataFrame(tfds.as_dataframe(raw_test_ds))
df_val = pd.DataFrame(tfds.as_dataframe(raw_val_ds))

# COMMAND ----------

# Nine most frequent emotions.
target_emotions = ['admiration','amusement','anger','annoyance','approval','curiosity','disapproval','gratitude','neutral']   
emotions = [e for e in df_train.columns if e != "comment_text"]
# Need mapping from emotion to indices of original and target emotions.
emotion_to_num = {emotion:i for i, emotion in enumerate(emotions)}
num_to_emotion = dict((v,k) for k,v in emotion_to_num.items())
target_emotion_nums = [emotion_to_num[emotion] for emotion in target_emotions]

# COMMAND ----------

def get_target_labels(df):
  # Converting from original 28 to 9 most frequent.
  label_data = df.drop("comment_text", axis=1)
  label_data = np.array(label_data)
  # Find index that contains the "1" which indicates emotion label.
  labels = list(label_data.argmax(axis=1))
  all_labels = labels
  # Filter to 9 most frequent.
  labels_idx = np.where(np.isin(labels, target_emotion_nums))[0]
  labels = [label for label in labels if label in target_emotion_nums]
  labels = np.array([target_emotions.index(num_to_emotion[num]) for num in labels])
  return labels, labels_idx

def get_target_text(df, idx):
  # Decode to string type and filter to target emotions.
  df_text = df["comment_text"].apply(lambda s : s.decode("utf-8"))
  df_text = df_text.iloc[idx].to_frame().reset_index(drop=True)
  return df_text

train_labels, train_labels_idx = get_target_labels(df_train)
test_labels, test_labels_idx = get_target_labels(df_test)
val_labels, val_labels_idx = get_target_labels(df_val)

train_text = get_target_text(df_train, train_labels_idx)
test_text = get_target_text(df_test, test_labels_idx)
val_text = get_target_text(df_val, val_labels_idx)

# Create larger train set by adding test set. 
test_train_text = pd.concat([train_text, test_text], ignore_index=True)
test_train_labels = np.append(train_labels, test_labels)
train_data_full = pd.DataFrame(data={"comment_text":test_train_text.comment_text.values, "label":test_train_labels})
train_data_full["emotion"] = train_data_full.label.apply(lambda x: target_emotions[x])
# Use val set for test set.
test_data_full = pd.DataFrame(data={"comment_text":val_text.comment_text.values, "label":val_labels})
test_data_full["emotion"] = test_data_full.label.apply(lambda x: target_emotions[x])

# COMMAND ----------

# To add an additional feature for training, we add a sentiment value to each text datapoint.
# This pipeline returns label : {positive, negative} and score : [0, 1.0]
# If the score is negative, we subtract it from 1. Else, we keep it.
# This allows for very negative posts to have a score ~0.0 and very positive ~1.0.
# This cell takes about 25 min to run.
pipe = pipeline("sentiment-analysis")
def add_sentiment_col(df):
    text_sentiments = []
    texts = list(df.comment_text.values)
    for text in texts:
        text_sentiments.append(pipe(text))
    text_scores = [sentiment[0]["score"] if sentiment[0]["label"] == "POSITIVE" else 1-sentiment[0]["score"] for sentiment in text_sentiments]
    df.insert(1, "sentiment", text_scores)
    return df

train_data = add_sentiment_col(train_data_full)
test_data = add_sentiment_col(test_data_full)

# COMMAND ----------

# Pick subset of four.
demo_emotions = ['amusement', 'annoyance', 'disapproval', 'neutral']
# For reproducibility
random_state = 8
def balance_neutral_class(df, random_state):
    data = df[df.emotion.isin(demo_emotions)]
    second_most_freq = data.emotion.value_counts().values[1]
    data.label = data['label'].map(lambda x: demo_emotions.index(target_emotions[x]))
    data_neutral = data[data.emotion == 'neutral']
    data_neutral = data_neutral.sample(n=second_most_freq, random_state=random_state)
    data_non_neutral = data[data.emotion != 'neutral']
    data_balanced = pd.concat([data_neutral, data_non_neutral], ignore_index=True)
    data_balanced = data_balanced.sort_values(by="comment_text")
    data_balanced = data_balanced.sample(frac=1, random_state=random_state)
    data_balanced = data_balanced.reset_index(drop=True)
    return data_balanced

train_data_balanced = balance_neutral_class(train_data, random_state)
test_data_balanced = balance_neutral_class(test_data, random_state)

# COMMAND ----------

dbutils.fs.rm("/tmp/emotionai/", recurse=True)
dbutils.fs.mkdirs("/tmp/emotionai/")
train_data_balanced.to_csv("/dbfs/tmp/emotionai/train_data_balanced.csv")
test_data_balanced.to_csv("/dbfs/tmp/emotionai/test_data_balanced.csv")

# COMMAND ----------



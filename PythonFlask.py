from flask import Flask, render_template, request
from flask_cors import CORS
import requests
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from official.nlp import optimization
from datasets import load_dataset
import nltk
from textaugment import EDA  #textaugment library for synonym replacement
import nlpaug.augmenter.word as naw  # NLPAug library for word augmenter by contextual word embedding

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
dataset = dataset["train"].train_test_split(test_size=0.3, seed=42)
aux_dataset = dataset["test"].train_test_split(test_size=0.33, seed=42)
dataset['validation'] = aux_dataset['test']
dataset['test'] = aux_dataset['train']
del (aux_dataset)
dataset["train"].to_csv(path + "intimacy/train.csv", index=False)
dataset["test"].to_csv(path + "intimacy/test.csv", index=False)
dataset["validation"].to_csv(path + "intimacy/validation.csv", index=False)
print("the three splits were saved into " + path + 'intimacy/')

dataset

app = Flask(__name__, static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

t = EDA()

# bert-base-multilingual-uncased Pretrained model on the top 102 languages with the largest Wikipedia using a masked language modeling (MLM) objective
aug = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased',
                                action="insert")


def generate(example):

  original_text = example['text']
  language = example['language']

  # textaugmenter
  if language == 'English':
    example['text_aug'] = t.synonym_replacement(original_text)
  else:
    example['text_aug'] = t.random_swap(original_text)

  #NLPAug
  example['text_nlpaug'] = aug.augment(original_text)[0]

  return example


ALL_DATA = True
if not ALL_DATA:
  sample = training_data.shuffle(seed=42).select(range(10))
  print(sample)
  sample = sample.map(generate)
  print(sample)
  for i in range(sample.num_rows):
    print('Original text:', sample[i]['text'])
    print('text augmented 1 (textaugmenter):', sample[i]['text_aug'])
    print('text augmented 2 (NLPaug):', sample[i]['text_nlpaug'])
    print()
  training_data = training_data.map(generate)

USE_DATA_AUGMENTED = True

models = [
  'bert-base-multilingual-uncased', 'cardiffnlp/twitter-xlm-roberta-base',
  'xlm-roberta-base', 'distilbert-base-multilingual-cased',
  'microsoft/Multilingual-MiniLM-L12-H384'
]
MODEL_NAME = models[1]  #0, 1, 2, 3, 4

print('Using model:', MODEL_NAME, USE_DATA_AUGMENTED)
from datasets import load_dataset, concatenate_datasets

dataset_name = "ISEGURA/mint"
access_token = "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"
LANGUAGES = set(dataset['train']['language'])

dataset

import re


def clean(examples):
  ## it applies the tokenzier on the dataset in its field text
  # we could add max_length = MAX_LENGHT, but in this case is not neccesary because MAX_LENTH is already 512, the maximum length allowed by the model
  new_texts = []
  for text in examples['text']:
    text = re.sub('@user', '', text)
    text = re.sub('http', '', text)
    text = re.sub('@[\w]+', '', text)
    text = text.strip()
    new_texts.append(text)

  examples['text'] = new_texts
  return examples


dataset = dataset.map(clean, batched=True)
dataset
from transformers import AutoTokenizer
if 'MiniLM' in MODEL_NAME:
  # we must load the tokenizer of XLM-R
  tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
else:
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

import pandas as pd

len_train_texts = [
  len(tokenizer(text).input_ids) for text in dataset['train']['text']
]
df = pd.Series(len_train_texts)
# free the space of this list
del (len_train_texts)
#show the statistics
df.describe(percentiles=[0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99])

MAX_LEN = 50


def tokenize(examples):
  ## it applies the tokenzier on the dataset in its field text
  # we could add max_length = MAX_LENGHT, but in this case is not neccesary because MAX_LENTH is already 512, the maximum length allowed by the model
  return tokenizer(examples["text"],
                   truncation=True,
                   max_length=MAX_LEN,
                   padding='max_length')


#apply tokenizer and remove the columns that we do not need anymore
data_encodings = dataset.map(tokenize,
                             batched=True,
                             remove_columns=['text', 'language'])
data_encodings

from transformers import AutoModelForSequenceClassification
# As num_labes is 1, the AutoModelForSequenceClassification will trigger the linear regression and use MSELoss() as the loss function automatically.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                           num_labels=1)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats


def compute_metrics_for_regression(eval_pred):
  logits, labels = eval_pred
  labels = labels.reshape(-1, 1)

  # loss metrics
  mse = mean_squared_error(labels, logits)
  rmse = mean_squared_error(labels, logits, squared=False)
  mae = mean_absolute_error(labels, logits)
  smape = 1 / len(labels) * np.sum(2 * np.abs(logits - labels) /
                                   (np.abs(labels) + np.abs(logits)) * 100)
  # performance metrics
  r2 = r2_score(labels, logits)
  pearson = stats.pearsonr(np.squeeze(np.asarray(labels)),
                           np.squeeze(np.asarray(logits)))
  pearson = pearson[0]
  # we return a dictionary with all metrics
  return {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "smape": smape,
    "pearson": pearson
  }
  # return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


from transformers import TrainingArguments

NUM_EPOCHS = 3  # paper used 15

# Specifiy the arguments for the trainer
training_args = TrainingArguments(
  output_dir='./results',
  num_train_epochs=NUM_EPOCHS,
  per_device_train_batch_size=64,  # 128 in the paper   
  per_device_eval_batch_size=20,
  weight_decay=0.01,
  learning_rate=2e-5,  # 0.001 in the paper,
  logging_dir='./logs',
  save_total_limit=10,
  load_best_model_at_end=True,
  # metric_for_best_model = 'rmse',
  metric_for_best_model='pearson',
  evaluation_strategy="epoch",  # steps in the paper
  save_strategy="epoch",  # steps in the paper
  report_to='all',
)
from transformers import Trainer

# Call the Trainer
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=data_encodings[
    'train'],  # if you only want to check the training is right, replace with train_dataset = data_encodings['train'].select(range(100))         
  eval_dataset=data_encodings[
    'validation'],  # if you only want to check the training is right, replace with eval_dataset = data_encodings['validation'].select(range(20)),                  
  compute_metrics=compute_metrics_for_regression,
  #callbacks=[EarlyStoppingCallback(3, 0.0)]
)

# Train the model
trainer.train()
trainer.evaluate()


def get_prediction(text):
  # prepare our text into tokenized sequence
  inputs = tokenizer(text,
                     max_length=MAX_LEN,
                     padding="max_length",
                     truncation=True,
                     return_tensors="pt").to("cuda")
  outputs = model(**inputs)  #output is a tensor
  return outputs[0].item(
  )  #we only have to return the value of the tensor by using item()


PATH_DATA = "/content/"
dataset_test = load_dataset("csv", data_files=PATH_DATA + "test_labeled.csv")
# clean the texts in the test dataset
# as we used for the texts in the training dataset
dataset_test = dataset_test.map(clean, batched=True)
dataset_test = dataset_test['train']
y_test = dataset_test['label']

# generate predictions for each text
y_pred = [get_prediction(text) for text in dataset_test['text']]

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
diff = [e1 - e2 for e1, e2 in zip(y_pred, y_test)
        ]  # Resultado: [-2, -1, -2, 0, -7, 6, 2]
smape = 1 / len(y_test) * np.sum(2 * np.abs(diff) /
                                 (np.abs(y_test) + np.abs(y_pred)) * 100)
# performance metrics
r2 = r2_score(y_test, y_pred)
pearson = stats.pearsonr(np.squeeze(np.asarray(y_test)),
                         np.squeeze(np.asarray(y_pred)))
pearson = pearson[0]

results = {
  'mse': mse,
  'rmse': rmse,
  'mae': mae,
  'smape': smape,
  'r2': r2,
  'pearson': pearson
}

import os

PATH = "/content/"
### Create an output directory
output_dir = PATH + 'results/'
if not os.path.exists(
    output_dir):  ### If the file directory doesn't already exists,
  os.makedirs(output_dir)  ### Make it please

# we use the test split to obtain final results
df = pd.DataFrame.from_dict(results.items())

# saving to csv
if '/' in MODEL_NAME:
  MODEL_NAME = MODEL_NAME[MODEL_NAME.index('/') + 1:]

path_results = output_dir + MODEL_NAME
if USE_DATA_AUGMENTED:
  path_results += '_aug'
path_results += '.csv'

df.to_csv(path_results, index=True)

print(path_results, ' was saved!')

test_dataset = pd.read_csv('semeval_test.csv')
test_dataset

counts_train = dataset['language'].value_counts().reset_index()
counts_train.columns = ['language', 'counts']
print('Languages in training: ', dataset['language'].unique())
print("Distribution in training:", counts_train)
print()
counts_test = test_dataset['language'].value_counts().reset_index()
counts_test.columns = ['language', 'counts']
print('Languages in test: ', test_dataset['language'].unique())
print("Distribution in test:", counts_test)
colors = [
  'deepskyblue', 'coral', 'olivedrab', 'blue', 'brown', 'purple', 'orange',
  'green', 'red', 'darkviolet'
]
labels = counts_test['language'].tolist()
print(labels)
dict_color = dict(zip(labels, colors))
dict_color
c_train = counts_train['language'].apply(lambda x: dict_color[x]).tolist()
c_test = counts_test['language'].apply(lambda x: dict_color[x]).tolist()


# No cacheing at all for API endpoints.
@app.after_request
def add_header(response):
  # response.cache_control.no_store = True
  if 'Cache-Control' not in response.headers:
    response.headers['Cache-Control'] = 'no-store'
  return response


CORS(app)


@app.route('/', methods=['GET', 'POST'])
def sendHomePage():
  if (request.method == "GET"):
    return render_template('index.html', notification='Welcome!')
  else:
    return render_template('index.html',
                           notification='You have given ' +
                           request.form['rating'] + ' Stars Feedback')


@app.route('/predict', methods=['POST'])
def PredictPossibility():
  GREScore = float(request.form['GREScore'])
  TOEFLScore = float(request.form['TOEFLScore'])
  UnivRating = float(request.form['UnivRating'])
  SOP = float(request.form['SOP'])
  LOR = float(request.form['LOR'])
  CGPA = float(request.form['CGPA'])
  return render_template('predict.html',
                         predict=probability,
                         comment=prob_comment,
                         color_scheme=color_scheme)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=81)


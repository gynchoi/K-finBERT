"""
2023-11-26
Rewrite /notebooks/finebert_training.ipynb to the python script by Gayoon Choi
"""

from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
from textblob import TextBlob
import argparse
import os

import nltk
nltk.download('punkt')

parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('--text_path', default="./data/sentiment_data/test.txt",type=str, help='Path to the text file.')
parser.add_argument('--output_dir', default="./data",type=str, help='Where to write the results')
parser.add_argument('--model_path', default="./models/classifier_model/finbert-sentiment",type=str, help='Path to classifier model')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

with open(args.text_path,'r',encoding='UTF8') as f:
    text = f.read()

model = AutoModelForSequenceClassification.from_pretrained(args.model_path,num_labels=3,cache_dir=None)

output = "predictions.csv"
result = predict(text, model, write_to_csv=True, path=os.path.join(args.output_dir,output))

blob = TextBlob(text)
result['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]
print(result.head())
print(f'Average sentiment is %.2f.' % (result.sentiment_score.mean()))
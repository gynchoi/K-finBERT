"""
2023-11-26
Transform AIHub dataset to finBERT/examples.csv form by Gayoon Choi

Orginal
    - format: json
    - dict {
        "annotation": [{
            "charNum":str, 
            "label":str, 
            "length":int, 
            "text":str, 
            "value":{"text_id": str, "word_num":int, "핵심동사":str, "확실성":str, "극성":str, "시제"str, "key_word":str}
            }], 
        "dupLabelYn" str,
        "metaData": {"ID":str, "FILENAME":str, "MEDIA_TYPE":str, "MEDIA_NAME":str, "CATEGORY":str, "PUBDATE":str, "SENTENCE":str}}

Transformed
    - format: csv or tsv
    - dict {
        "id": int[optional],
        "text": str,
        "label: str
    }
                
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import json
import glob
import csv
import os

parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('--data_path', default="/home/guest/workspace/K-finBERT/data/sentiment_data/finance_aug/finance_augmented.csv", type=str, help='Path to the text file.')

def make_csv(data_path):
    with open(data_path, 'r', encoding='UTF8') as f:
        json_data= json.load(f)

    id = json_data["metaData"]["ID"]
    values = json_data["annotation"]

    for val in values:
        text = val["text"]
        key = val["value"]["극성"]
        label_dict = {"긍정": 'positive', "부정": 'negative', "미정": 'neutral'}

        wr.writerow([id, text, label_dict[key]])
    
def make_text(data_path):
    with open(data_path, 'r', encoding='UTF8') as f:
        json_data= json.load(f)

    values = json_data["annotation"]
    for val in values:
        text = val["text"]

        wr.writerow([text])  

def translate(data_path):
    csv_data = pd.read_csv(data_path, encoding='UTF8')

    df = pd.DataFrame()
    df.index.name = 'id'
    df['text'] = csv_data['kor_sentence']
    df['label'] = csv_data['labels']
    df.to_csv('./data/sentiment_data/finance/finance.csv')

    return df

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists('./data/sentiment_data'):
        os.makedirs('./data/sentiment_data')

    if 'zip' in args.data_path:
        paths = glob.glob(os.path.join(args.data_path, "*.json"))

        if "TL" in args.data_path:
            phase = "train"
        elif "VL" in args.data_path:
            phase = "validation"

        f = open(f'./data/sentiment_data/aihub/{phase}.csv', 'w', encoding='utf-8', newline='')
        wr = csv.writer(f, delimiter='\t')
        wr.writerow(["id", "text","label"])

        for path in paths:
            make_csv(path)
        f.close()

        print(f"Finish: sentiment_data/aihub/{phase}.csv")

        if phase == "validation":
            f = open(f'./data/sentiment_data/aihub/test.txt', 'w', encoding='utf-8', newline='')
            wr = csv.writer(f, delimiter='\t')

            for path in paths:
                make_text(path)
            f.close()

            print(f"Additional file: sentiment_data/aihub/test.txt")

    elif 'finance' in args.data_path:
        data = translate(args.data_path)

        train, test = train_test_split(data, test_size=0.2, random_state=0)
        train, valid = train_test_split(train, test_size=0.1, random_state=0)
        train.to_csv('data/sentiment_data/finance_aug/train.csv',sep='\t')
        test.to_csv('data/sentiment_data/finance_aug/test.csv',sep='\t')
        valid.to_csv('data/sentiment_data/finance_aug/validation.csv',sep='\t')

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

import argparse
import json
import glob
import csv
import os

parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('--data_path', default="C:/Users/gynch/workspace/TL_뉴스_금융.zip", type=str, help='Path to the text file.')

def read_and_write(data_path):
    with open(data_path, 'r', encoding='UTF8') as f:
        json_data= json.load(f)

    id = json_data["metaData"]["ID"]
    values = json_data["annotation"]

    for val in values:
        text = val["text"]
        key = val["value"]["극성"]
        label_dict = {"긍정": 'positive', "부정": 'negative', "미정": 'neutral'}

        wr.writerow([id, text, label_dict[key]])
    

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists('./data/sentiment_data'):
        os.makedirs('./data/sentiment_data')

    paths = glob.glob(os.path.join(args.data_path, "*.json"))

    if "TL" in args.data_path:
        phase = "train"
    elif "VL" in args.data_path:
        phase = "validation"

    f = open(f'./data/sentiment_data/{phase}.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f, delimiter='\t')
    wr.writerow(["id", "text","label"])

    for path in paths:
        read_and_write(path)

    f.close()

    print(f"Finish: sentiment_data/{phase}.csv")


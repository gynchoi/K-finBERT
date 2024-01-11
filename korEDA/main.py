"""
2023-12-06
Augment positive/negative dataset with EDA

Reference
- EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks (https://arxiv.org/abs/1901.11196)
- Korean EDA: https://github.com/catSirup/KorEDA

"""
from sklearn.model_selection import train_test_split
import pandas as pd

import argparse
import csv

from eda import EDA

parser = argparse.ArgumentParser(description='EDA Augmentation')
parser.add_argument('--sr', action="store_true", help="apply sr augmentation")
parser.add_argument('--ri', action="store_true", help="apply ri augmentation")
parser.add_argument('--rs', action="store_true", help="apply rs augmentation")
parser.add_argument('--rd', action="store_true", help="apply rd augmentation")
parser.add_argument('-p', '--positive', action="store_true", help="apply positive augmentation")
parser.add_argument('-n', '--negative', action="store_true", help="apply negative augmentation")
parser.add_argument('--num_p', default=1, type=str, help="number of augmention in positive label")
parser.add_argument('--num_n', default=4, type=str, help="number of augmention in negative label")
parser.add_argument('--load_path', default="/home/guest/workspace/K-finBERT/data/sentiment_data/finance/train.csv", type=str, help='Path to the text file.')
parser.add_argument('--save_path', default="/home/guest/workspace/K-finBERT/data/sentiment_data/finance_aug", type=str, help='Path to the text file.')

def main(args):
    df = pd.DataFrame()

    f = open(args.load_path, 'r', encoding='utf-8') 
    reader = csv.reader(f, delimiter="\t")

    for i, line in enumerate(reader):
        if i>0:
            id, text, label = line[0], line[1], line[2]
            if label=="positive" and args.positive:
                augmented_texts = EDA(sentence=text, num_aug=args.num_p, args=args)
                for text in augmented_texts:
                    df = df.append({"id": id, "text": text, "label": label}, ignore_index=True)
            elif label=="negative" and args.negative:
                augmented_texts = EDA(sentence=text, num_aug=args.num_n, args=args)
                for text in augmented_texts:
                    df = df.append({"id": id, "text": text, "label": label}, ignore_index=True)
            else:
                df = df.append({"id": id, "text": text, "label": label}, ignore_index=True)

    f.close()

    print('--------After Augmentation-----------')
    print(f'Neutral: {df[df["label"] == "neutral"].shape[0]} ({round(df[df["label"] == "neutral"].shape[0]/len(df) * 100,3)}%)')
    print(f'Positive: {df[df["label"] == "positive"].shape[0]} ({round(df[df["label"] == "positive"].shape[0]/len(df) * 100,3)}%)')
    print(f'Negative: {df[df["label"] == "negative"].shape[0]} ({round(df[df["label"] == "negative"].shape[0]/len(df) * 100,3)}%)')

    df.to_csv(f"{args.save_path}/train.csv",sep='\t',index=False)

if __name__ == "__main__":
    args = parser.parse_args()

    print('--------Before Augmentation-----------')
    df = pd.read_csv(args.load_path, encoding='UTF8', sep="\t")
    print(f'Neutral: {df[df["label"] == "neutral"].shape[0]} ({round(df[df["label"] == "neutral"].shape[0]/len(df) * 100,3)}%)')
    print(f'Positive: {df[df["label"] == "positive"].shape[0]} ({round(df[df["label"] == "positive"].shape[0]/len(df) * 100,3)}%)')
    print(f'Negative: {df[df["label"] == "negative"].shape[0]} ({round(df[df["label"] == "negative"].shape[0]/len(df) * 100,3)}%)')

    main(args)

# K-FinBERT: Korean Financial Sentiment Analysis with BERT
This repository is fine-tuning FinBERT with Korean financial sentence dataset from AIHub. For detailed information about FinBERT, see the [FinBERT](https://github.com/ProsusAI/finBERT).

Also, you can view learning curves in the [wandb](https://wandb.ai/gynchoi17/K-finBERT/overview?workspace=user-gynchoi17)

## Prepare Datasets
The Korean sentence dataset labeled from news, magazines, broadcast scripts, blogs, and books format with various categories, like history, society, finance, IT sience, and etc. Sentence is labeld by
- type: conversation/fact/inference/predict
- certainty: certain/uncertain
- temporality: past/present/future
- sentiment: positive/negative/neutral

We only use the finance news dataset for training.

**Download**\
Download the '문장 유형(추론, 예측 등) 판단 데이터' dataset from [AIHub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71486). As the dataset is split compressed, a program that supports split decompression (e.g. Bandizip, 7-Zip) is recommended.

**Preprocess**\
Dataset should be preprocessed to {'ID': int, 'Text': str, 'Label': str} form and saved as csv with delimiter='\t'. 
```bash
python ./scripts/data_utils.py --data_path "your_directory/TL_뉴스_금융.zip" # not zip file but folder
python ./scripts/data_utils.py --data_path "your_directory/VL_뉴스_금융.zip"
```


## Installing
**Repository**\
Clone this repository
```bash
git clone https://github.com/gynchoi/finBERT.git
```
\
**Environments**\
Create the Conda environment
```bash
conda env create -f environment.yml
conda activate finbert
```

**Pre-trained model Checkpoints**\
Download the original FinBERT checkpoint from [HuggingFace/FinBERT](https://huggingface.co/ProsusAI/finbert). 
```bash
mkdir models/sentiment
cd ./models/sentiment/

git lfs install
git clone https://huggingface.co/ProsusAI/finbert
```
If you got `git: 'lfs' is not a git command. See 'git --help` error, install git-lfs in your terminal first.
```bash
sudo apt install git-lfs
```
## Minor Modifications
**Tokenizer**\
For utilizing the Korean dataset, tokenizer is changed from 'bert-base-uncased' to 'monologg/kobert'. To use KoBERT tokenizer
1. Copy [tokenization_kobert.py](https://github.com/monologg/KoBERT-Transformers/blob/master/kobert_transformers/tokenization_kobert.py) to ./finbert/ folder
2. Download sentencepiece package, unless you may get `UnboundLocalError: local variable 'spm' referenced before assignment` error
    ```bash
    pip install sentencepiecc
    ```
3. Modify the './finbert/finbert.py' code
    ```python
    from finbert.tokenization_kobert import KoBertTokenizer

    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # remove this code
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

    ...

    # self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, do_lower_case=self.config.do_lower_case) # remove this code
    self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    ```
\
**Encoding**\
When open Korean dataset, encoding is needed. Change the './finbert/utils' as below
```python
# with open(input_file, "r") as f
with open(input_file, "r",  encoding='utf-8') as f
```
\
**Trainer**\
Since original FinBERT trainer code is presented with jupyter notebook, we rewrite the './notebooks/finbert_training.ipynb' to python format
1. Joining paths in 'finbert.py' with OS package
```python
# self.config.model_dir / ('temporary' + str(best_model)
import os

os.path.join(self.config.model_dir, ('temporary' + str(best_model)))
```

**Test**\
From predict.py, we need to download and import nltk 
```python
import nltk
nltk.download('punkt')
```

## Errors Report
You can view expected errors and some solutions in [ERRORS.md](https://github.com/gynchoi/K-finBERT/blob/master/ERRORS.md)
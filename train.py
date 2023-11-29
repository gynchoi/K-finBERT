"""
2023-11-26
Rewrite /notebooks/finebert_training.ipynb to the python script by Gayoon Choi
"""

from pathlib import Path
import argparse
import shutil
import logging
import sys
sys.path.append('..')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import wandb

from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification

from finbert.finbert import *
from finbert.seed import fix_seed

parser = argparse.ArgumentParser(description='Sentiment analyzer')
# Fine tuning
parser.add_argument('--partial', action="store_true")
parser.add_argument('--epochs', default=4, type=int, help="number of epochs for training")
# Paths
parser.add_argument('--lm_path', default="./models/sentiment/finbert",type=str, help='The BERT model to be used')
parser.add_argument('--cl_path', default="./models/classifier_model/finbert-sentiment",type=str, help='The path where the resulting model will be saved')
parser.add_argument('--cl_data_path', default="./data/sentiment_data/",type=str, help='Path to the text file.')
# Wandb
parser.add_argument('--name', default=None, type=str, help='Wandb run name')

def configure_training(args):
    # Configuring training parameters
    try:
        shutil.rmtree(args.cl_path) 
    except:
        pass

    bertmodel = AutoModelForSequenceClassification.from_pretrained(args.lm_path,cache_dir=None, num_labels=3)


    config = Config(data_dir=args.cl_data_path,
                    bert_model=bertmodel,
                    num_train_epochs=args.epochs,
                    model_dir=args.cl_path,
                    max_seq_length = 48,
                    train_batch_size = 32,
                    learning_rate = 2e-5,
                    output_mode='classification',
                    warm_up_proportion=0.2,
                    local_rank=-1,
                    discriminate=True,
                    gradual_unfreeze=True)

    finbert = FinBert(config)
    # finbert.base_model = 'bert-base-uncased'
    finbert.base_model = 'monologg/kobert'
    # finbert.base_model = 'bert-base-multilingual-cased'
    finbert.config.discriminate=True
    finbert.config.gradual_unfreeze=True
    finbert.prepare_model(label_list=['positive','negative','neutral'])

    return finbert


def subsettuning(model):
    # [Optional] Fine-tuning a subset of the model
    freeze = 6

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
        
    for i in range(freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    return model

def training(finbert, args):
    train_data = finbert.get_data('train')
    model = finbert.create_the_model()

    if args.partial:
        model = subsettuning(model)

    # Training
    trained_model = finbert.train(train_examples = train_data, model = model)

    return trained_model

def eval(finbert, trained_model):
    # Test
    test_data = finbert.get_data('test')
    results = finbert.evaluate(examples=test_data, model=trained_model)

    return results


def report(df, cols=['label','prediction','logits']):
    # Classification report
    #print('Validation loss:{0:.2f}'.format(metrics['best_validation_loss']))
    cs = CrossEntropyLoss(weight=finbert.class_weights)
    loss = cs(torch.tensor(list(df[cols[2]])),torch.tensor(list(df[cols[0]])))
    print("Loss:{0:.2f}".format(loss))
    print("Accuracy:{0:.2f}".format((df[cols[0]] == df[cols[1]]).sum() / df.shape[0]) )
    print("\nClassification Report:")
    print(classification_report(df[cols[0]], df[cols[1]]))



if __name__ == "__main__":
    args = parser.parse_args()

    # Wandb logging
    if args.name is None:
        args.name = f'epoch_{args.epochs}'
    wandb.init(project='K-finBERT', name=args.name, entity="gynchoi17")

    # Modules
    project_dir = Path.cwd().parent
    pd.set_option('max_colwidth', -1)
    fix_seed(42)

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.ERROR)
    
    finbert = configure_training(args)
    
    trained_model = training(finbert, args)
    results = eval(finbert, trained_model)

    results['prediction'] = results.predictions.apply(lambda x: np.argmax(x,axis=0))
    report(results,cols=['labels','prediction','predictions'])
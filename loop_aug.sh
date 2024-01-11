#!/bin/bash

tokenizers=('klue/bert-base' 'monologg/kobigbird-bert-base' 'bert-base-multilingual-cased' 'monologg/kobert')
sizes=(16 32 64)
lrs=('1e-5' '2e-5' '3e-5' '4e-5' '5e-5')
warmups=(0.0 0.1 0.2 0.3) 

for tokenizer in "${tokenizers[@]}"; do
    for size in "${sizes[@]}"; do
        for lr in "${lrs[@]}"; do
            for warmup in "${warmups[@]}"; do
                python3 -u ./train.py \
                    --epochs 15 \
                    --tokenizer ${tokenizer} \
                    --batch_size ${size} \
                    --lr ${lr} \
                    --warmup ${warmup} \
                    --cl_data_path "/home/guest/workspace/K-finBERT/data/sentiment_data/finance_aug" \
                    --name "${tokenizer}_${size}_${lr}_${warmup}" \
                    --project "Aug_K-finBERT"
            done
        done
    done
done
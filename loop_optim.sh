#!/bin/bash

tokenizers=('klue/bert-base')
sizes=(16 32 64)
lrs=('1e-7' '2e-7' '1e-6' '2e-6' '1e-5' '2e-5' '3e-5' '4e-5' '5e-5')
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
                    --name "${tokenizer}_${size}_${lr}_${warmup}"
            done
        done
    done
done
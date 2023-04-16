#!/bin/bash
# train baseline
DATA=/root/khang/data/para/de2en/bin/baseline/
SAVE_DIR=checkpoints/tmp

fairseq-train --fp16 \
 --task translation_from_pretrained_xlm \
 --source-lang de --target-lang en \
 --upsample-primary 1 \
 $DATA --combine-val \
 --save-dir $SAVE_DIR \
 --arch transformer --share-decoder-input-output-embed \
 --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --encoder-attention-heads 8 \
 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 --decoder-attention-heads 8 \
 --dropout 0.3 \
 --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
 --lr 5e-4 --lr-scheduler inverse_sqrt \
 --warmup-updates 4000 --warmup-init-lr 1e-07 --weight-decay 0.0001 \
 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
 --batch-size 16 --update-freq 8 \
 --max-epoch 25 --no-epoch-checkpoints \
 --eval-bleu \
 --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
 --eval-bleu-detok moses \
 --eval-bleu-remove-bpe --eval-bleu-print-samples \
 --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
 #--wandb-project $WANDB

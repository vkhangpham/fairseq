CUDA_VISIBLE_DEVICES=2 fairseq-train \
data-bin/iwslt14_80k \
--arch transformer --share-decoder-input-output-embed \
--encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
--save-dir checkpoints/80k/finetune --max-epoch 15 --no-epoch-checkpoints \
--optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
--lr 5e-4 --warmup-updates 4000 --lr-scheduler reduce_lr_on_plateau \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 4096 \
--eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
--eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--finetune-from-model checkpoints/placeholder/checkpoint_best.pt \
--wandb-project sf-80k --memory-efficient-fp16

cp checkpoints/80k/finetune/checkpoint_last.pt checkpoints/80k/finetune/checkpoint_15.pt

CUDA_VISIBLE_DEVICES=2 fairseq-train \
data-bin/iwslt14_80k \
--arch transformer --share-decoder-input-output-embed \
--encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
--save-dir checkpoints/80k/finetune --max-epoch 50 --no-epoch-checkpoints \
--optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
--lr 5e-4 --warmup-updates 4000 --lr-scheduler reduce_lr_on_plateau \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 4096 \
--eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
--eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--wandb-project sf-80k --memory-efficient-fp16
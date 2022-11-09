CUDA_VISIBLE_DEVICES=($1) fairseq-train \
data-bin/iwslt14_80k \
--arch transformer --share-decoder-input-output-embed \
--encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
--save-dir checkpoints/placeholder --max-epoch 1 \
--optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
--lr 1e-9 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 4096 \
--eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
--eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--fp16
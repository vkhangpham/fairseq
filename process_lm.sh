TEXT=data-bin/80k/tokenized
fairseq-preprocess \
    --only-source \
    --srcdict data-bin/80k/dict.txt \
    --trainpref $TEXT/train.all \
    --validpref $TEXT/valid.all \
    --testpref $TEXT/test.all \
    --destdir data-bin/80k/decoder \
    --workers 20
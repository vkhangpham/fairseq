import os
os.system("fairseq-train data-bin/iwslt14.tokenized.de-en --arch lstm_dummy --optimizer adam --lr 0.005 --lr-shrink 0.5 --max-tokens 10000")
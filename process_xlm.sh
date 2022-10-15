DATA=/root/khang/code/XLM/data/processed/15M_60k
OUTPATH=/root/khang/code/fairseq/data-bin/15M_60k
mkdir -p "$OUTPATH"

for lg in de en
do

	  fairseq-preprocess \
		    --task cross_lingual_lm \
		      --srcdict $DATA/vocab \
		        --only-source \
			  --trainpref $DATA/train \
			    --validpref $DATA/valid \
			      --testpref $DATA/test \
			        --destdir $OUTPATH \
				  --workers 20 \
				    --source-lang $lg

	    for stage in train test valid;do
		        sudo mv "$OUTPATH/$stage.$lg-None.$lg.bin" "$OUTPATH/$stage.$lg.bin"
			    sudo mv "$OUTPATH/$stage.$lg-None.$lg.idx" "$OUTPATH/$stage.$lg.idx"
			      done
		      done

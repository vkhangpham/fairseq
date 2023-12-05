# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lstm_dapt.lstm_dapt import LSTMDAPTModel
import argparse

CNN_KWARGS = dict(beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)


@torch.no_grad()
def generate(dapt, infile, outfile="dapt_hypo.txt", bsz=32, n_obs=None, **eval_kwargs):
    count = 1

    # if n_obs is not None: bsz = min(bsz, n_obs)

    with open(infile) as source, open(outfile, "w") as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if n_obs is not None and count > n_obs:
                break
            if count % bsz == 0:
                hypotheses_batch = dapt.sample(slines, **eval_kwargs)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + "\n")
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1

        if slines != []:
            hypotheses_batch = dapt.sample(slines, **eval_kwargs)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + "\n")
                fout.flush()


def main():
    """
    Usage::

         python examples/bart/summarize.py \
            --model-dir $HOME/bart.large.cnn \
            --model-file model.pt \
            --src $HOME/data-bin/cnn_dm/test.source
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--model-file",
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--src", default="test.source", help="text to summarize", type=str
    )
    parser.add_argument(
        "--out", default="test.hypo", help="where to save summaries", type=str
    )
    parser.add_argument("--bsz", default=32, help="where to save summaries", type=int)
    parser.add_argument(
        "--n", default=None, help="how many examples to summarize", type=int
    )
    args = parser.parse_args()
    eval_kwargs = CNN_KWARGS
    if args.model_dir == "pytorch/fairseq":
        dapt = torch.hub.load("pytorch/fairseq", args.model_file)
    else:
        dapt = LSTMDAPTModel.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_file,
            data_name_or_path=args.model_dir,
        )
    dapt = dapt.eval()
    if torch.cuda.is_available():
        dapt = dapt.cuda().half()
    generate(
        dapt, args.src, bsz=args.bsz, n_obs=args.n, outfile=args.out, **eval_kwargs
    )


if __name__ == "__main__":
    main()
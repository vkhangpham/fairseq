# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def compute_cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Function to compute the cross entropy loss. The default value of
    ignore_index is the same as the default value for F.cross_entropy in
    pytorch.
    """
    assert logits.size(0) == targets.size(
        -1
    ), "Logits and Targets tensor shapes don't match up"

    loss = F.nll_loss(
        F.log_softmax(logits, -1, dtype=torch.float32),
        targets,
        reduction="sum",
        ignore_index=ignore_index,
    )
    return loss


@register_criterion("mlm_mse")
class MLM_MSELoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    This optionally also computes the next sentence prediction (NSP) loss and
    adds it to the overall loss based on the specified args. There are three
    cases to consider:
        1) Generic MLM training without NSP loss. In this case sentence_targets
           and sentence_logits are both None.
        2) BERT training without NSP loss. In this case sentence_targets is
           not None but sentence_logits is None and we should not be computing
           a sentence level loss.
        3) BERT training with NSP loss. In this case both sentence_targets and
           sentence_logits are not None and we should be computing a sentence
           level loss. The weight of the sentence level loss is specified as
           an argument.
    """

    def __init__(self, task, masked_lm_only, nsp_loss_weight=False):
        super().__init__(task)
        self.masked_lm_only = masked_lm_only
        self.nsp_loss_weight = nsp_loss_weight

    @staticmethod
    def add_args(parser):
        """Args for MaskedLM Loss"""
        parser.add_argument(
            "--masked-lm-only",
            default=True,
            action="store_true",
            help="compute MLM loss only",
        )
        parser.add_argument(
            "--nsp-loss-weight",
            default=1.0,
            type=float,
            help="weight for next sentence prediction" " loss (default 1)",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # torch.autograd.set_detect_anomaly(True)
        try:
            #print (sample['net_input'])
            # _, _ = model(**sample["net_input"])
            # return torch.tensor(1., requires_grad=True), 1, {}
            # with torch.autograd.detect_anomaly():
            # lm_logits, output_metadata = model(**sample["net_input"])
            #
            # # reshape lm_logits from (N,T,C) to (N*T,C)
            # lm_logits = lm_logits.view(-1, lm_logits.size(-1))
            # lm_targets = sample["lm_target"]
            # masked_tokens = lm_targets.ne(self.padding_idx)
            # lm_targets = lm_targets.view(-1)
            # lm_loss = compute_cross_entropy_loss(lm_logits, lm_targets, self.padding_idx)

            masked_tokens = sample["lm_target"].ne(self.padding_idx)
            # sample_size = masked_tokens.int().sum()
            if masked_tokens.device == torch.device("cpu"):
                if not masked_tokens.any():
                    masked_tokens = None
            else:
                masked_tokens = torch.where(
                    masked_tokens.any(),
                    masked_tokens,
                    masked_tokens.new([True]),
                )
            output = model(**sample["net_input"], masked_tokens=masked_tokens)
            logits = output[0]
            targets = sample["lm_target"]
            if masked_tokens is not None:
                targets = targets[masked_tokens]
            # compute the number of tokens for which loss is computed. This is used
            # to normalize the loss
            # ntokens = utils.strip_pad(targets, self.padding_idx).numel()
            ntokens = masked_tokens.int().sum()
            lm_loss = modules.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum",
                ignore_index=self.padding_idx,
            )

            # Get weights of the final linear layer
            linear_weights = model.encoder.embed_out.weight.data
            linear_weights = linear_weights[targets]

            # Get final hidden embeddings (before LN)
            final_hidden = output[1]['final_hidden']
            # final_hidden = final_hidden.reshape(-1, final_hidden.size(-1))
            # final_hidden = final_hidden[masked_tokens]

            # Calculate MSE loss between linear weights and final hidden embeddings
            mse_loss = F.mse_loss(
                final_hidden,
                linear_weights,
                reduction='none'
            )
            mse_loss = mse_loss.mean(-1).sum()
            # mse_loss = mse_loss / ntokens
            # mse_loss.data = mse_loss.data * ntokens
            loss = (lm_loss+ mse_loss)/ntokens
            nsentences = sample["nsentences"]
            sample_size = ntokens
            logging_output = {
                "loss": utils.item(loss.data) if reduce else loss.data,
                "lm_loss": utils.item(lm_loss.data) if reduce else lm_loss.data,
                "mse_loss": mse_loss.data,
                # sentence loss is not always computed
                "sentence_loss": 0.0,
                "ntokens": ntokens,
                "nsentences": nsentences,
                "sample_size": sample_size,
            }
            # from IPython import embed; embed()
            return loss, sample_size, logging_output
        except Exception as e:
            import os
            print(f"Error in {os.path.basename(__file__)}")
            from IPython import embed; embed()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        lm_loss_sum = sum(log.get("lm_loss", 0) for log in logging_outputs)
        mse_loss_sum = sum(log.get("mse_loss", 0) for log in logging_outputs)
        sentence_loss_sum = sum(log.get("sentence_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_loss = sum(log.get("loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss",
            agg_loss / sample_size / math.log(2) if sample_size > 0 else 0.0,
            sample_size,
            round=3,
        )
        metrics.log_scalar(
            "mlm_loss",
            lm_loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.0,
            ntokens,
            round=3,
        )
        metrics.log_scalar(
            "mse_loss",
            mse_loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.0,
            ntokens,
            round=3,
        )
        metrics.log_scalar(
            "sentence_loss",
            sentence_loss_sum / nsentences / math.log(2) if nsentences > 0 else 0.0,
            nsentences,
            round=3,
        )
        metrics.log_scalar(
            "nll_loss",
            lm_loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.0,
            ntokens,
            round=3,
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by forward can be summed
        across workers prior to calling reduce_metrics. Setting this
        to True will improves distributed training speed.
        """
        return True
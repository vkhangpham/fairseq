# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP,
    Embedding,
    TransformerDecoder,
)
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from fairseq.utils import safe_getattr, safe_hasattr

import torch

DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class CLDecoderConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    decoder_embed_dim: int = field(
        default=1024, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=4096, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    character_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, uses character embedding convolutions to produce token embeddings"
        },
    )
    character_filters: str = field(
        default="[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
        metadata={"help": "size of character embeddings"},
    )
    character_embedding_dim: int = field(
        default=4, metadata={"help": "size of character embeddings"}
    )
    char_embedder_highway_layers: int = field(
        default=2,
        metadata={"help": "number of highway layers for character token embeddder"},
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    adaptive_input_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    adaptive_input_cutoff: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of adaptive input cutoff points."},
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        },
    )
    # config for "BASE Layers: Simplifying Training of Large, Sparse Models"
    base_layers: Optional[int] = field(
        default=0, metadata={"help": "number of BASE layers in total"}
    )
    base_sublayers: Optional[int] = field(
        default=1, metadata={"help": "number of sublayers in each BASE layer"}
    )
    base_shuffle: Optional[int] = field(
        default=1,
        metadata={"help": "shuffle tokens between workers before computing assignment"},
    )
    # NormFormer
    scale_fc: Optional[bool] = field(
        default=False,
        metadata={"help": "Insert LayerNorm between fully connected layers"},
    )
    scale_attn: Optional[bool] = field(
        default=False, metadata={"help": "Insert LayerNorm after attention"}
    )
    scale_heads: Optional[bool] = field(
        default=False,
        metadata={"help": "Learn a scale coefficient for each attention head"},
    )
    scale_resids: Optional[bool] = field(
        default=False,
        metadata={"help": "Learn a scale coefficient for each residual connection"},
    )

    # xFormers arguments
    decoder_xformers_att_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "config for xFormers library attention, defined in xformers.components.attention.AttentionConfig",
        },
    )

    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")


@register_model("cl_decoder", dataclass=CLDecoderConfig)
class CLDecoder(FairseqLanguageModel):
    def __init__(self, decoder, interface):
        super().__init__(decoder)
        self.interface = interface
        self.bpe_end = []

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if safe_getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = safe_getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = TransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=False
        )

        interface = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )
        interface.load_state_dict(torch.load("/root/khang/checkpoints/emb.pt"))
        interface.requires_grad_(False)
        # if args.interface_path:
        #     interface.load_state_dict(torch.load(interface_path))

        return cls(decoder, interface)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens

    def add_noise(self, src_tokens, src_lengths):
        """
        Add noise to the encoder input.
        """
        dico = self.decoder.dictionary.symbols
        bpe_ends = np.array([not dico[i].endswith('@@') for i in range(len(dico))])

        src_tokens, src_lengths = word_shuffle(bpe_ends, src_tokens, src_lengths)
        src_tokens, src_lengths = word_dropout(dico, bpe_ends, src_tokens, src_lengths)
        return src_tokens, src_lengths


    def forward(self, src_tokens, **kwargs):
        # noisy_sample = self.add_noise(src_tokens, src_lengths)
        encoder_out = self.interface(src_tokens)
        # B x T x C -> T x B x C
        encoder_out = encoder_out.transpose(0, 1)
        encoder_out = {
            "encoder_out": [encoder_out],  # T x B x C
            "encoder_padding_mask": [src_tokens.eq(self.decoder.padding_idx)]
        }
        return self.decoder(src_tokens, encoder_out)

def word_shuffle(bpe_ends, x, l, word_shuffle=3):
    """
    Randomly shuffle input words.
    """
    if word_shuffle == 0:
        return x, l
    from IPython import embed
    print("In word_shuffle")
    embed()
    # define noise word scores
    noise = np.random.uniform(0, word_shuffle, size=(x.size(0) - 1, x.size(1)))
    noise[0] = -1  # do not move start sentence symbol

    # be sure to shuffle entire words
    bpe_end = bpe_ends[x]
    word_idx = bpe_end[::-1].cumsum(0)[::-1]
    word_idx = word_idx.max(0)[None, :] - word_idx

    assert word_shuffle > 1
    x2 = x.clone()
    for i in range(l.size(0)):
        # generate a random permutation
        scores = word_idx[:l[i] - 1, i] + noise[word_idx[:l[i] - 1, i], i]
        scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
        permutation = scores.argsort()
        # shuffle words
        x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
    return x2, l

def word_dropout(dictionary, bpe_ends, x, l, word_dropout=.1):
    """
    Randomly drop input words.
    """
    if word_dropout == 0:
        return x, l
    assert 0 < word_dropout < 1

    # define words to drop
    bos_index = bos_index[lang_id]
    keep = np.random.rand(x.size(0) - 1, x.size(1)) >= word_dropout
    keep[0] = 1  # do not drop the start sentence symbol

    # be sure to drop entire words
    bpe_end = bpe_ends[x]
    word_idx = bpe_end[::-1].cumsum(0)[::-1]
    word_idx = word_idx.max(0)[None, :] - word_idx

    sentences = []
    lengths = []
    for i in range(l.size(0)):
        assert x[l[i] - 1, i] == dictionary.eos()
        words = x[:l[i] - 1, i].tolist()
        # randomly drop words from the input
        new_s = [w for j, w in enumerate(words) if keep[word_idx[j, i], i]]
        # we need to have at least one word in the sentence (more than the start / end sentence symbols)
        if len(new_s) == 1:
            new_s.append(words[np.random.randint(1, len(words))])
        new_s.append(dictionary.eos())
        assert len(new_s) >= 3 and new_s[0] == bos_index and new_s[-1] == dictionary.eos()
        sentences.append(new_s)
        lengths.append(len(new_s))
    # re-construct input
    l2 = torch.LongTensor(lengths)
    x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(dictionary.pad())
    for i in range(l2.size(0)):
        x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
    return x2, l2

def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if safe_hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if safe_hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.0)

    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = safe_getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = safe_getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = safe_getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.activation_fn = safe_getattr(args, "activation_fn", "relu")

    args.decoder_layerdrop = safe_getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = safe_getattr(args, "decoder_layers_to_keep", None)
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    args.base_layers = safe_getattr(args, "base_layers", 0)
    args.base_sublayers = safe_getattr(args, "base_sublayers", 1)
    args.base_shuffle = safe_getattr(args, "base_shuffle", False)

    args.add_bos_token = safe_getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = safe_getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.character_embeddings = safe_getattr(args, "character_embeddings", False)

    args.decoder_output_dim = safe_getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = safe_getattr(
        args, "decoder_input_dim", args.decoder_embed_dim
    )

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = safe_getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = safe_getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = safe_getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = safe_getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = safe_getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = safe_getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = safe_getattr(args, "offload_activations", False)
    args.scale_fc = safe_getattr(args, "scale_fc", False)
    args.scale_attn = safe_getattr(args, "scale_attn", False)
    args.scale_heads = safe_getattr(args, "scale_heads", False)
    args.scale_resids = safe_getattr(args, "scale_resids", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("cl_decoder", "cl_decoder_base")
def transformer_lm_big(args):
    args.decoder_layers = safe_getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 8)
    base_lm_architecture(args)
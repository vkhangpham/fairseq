# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import defaultdict
from dataclasses import dataclass, field
import os
import random
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
from fairseq.data import UnsupervisedMTNoising

import torch
import numpy as np    
import fastBPE
from itertools import permutations

DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class XLITDecoderConfig(FairseqDataclass):
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

    # XLIT Encoder
    semface_path: Optional[str] = field(
        default="",
        metadata={"help": "path to trained semantic interface"},
    )
    reload_xlit_encoder_emb: Optional[bool] = field(
        default=False,
        metadata={"help": "Freeze decoder embeddings."},
    )
    bpe_codes: Optional[str] = field(
        default="",
        metadata={"help": "path to bpe codes"},
    )
    bpe_vocab: Optional[str] = field(
        default="",
        metadata={"help": "path to bpe vocab"},
    )
    biling_dict_path: Optional[str] = field(
        default="",
        metadata={"help": "path to bilingual dictionaries"},
    )
    

    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")


@register_model("xlit_decoder", dataclass=XLITDecoderConfig)
class XLITDecoder(FairseqLanguageModel):
    def __init__(self, decoder, interface, args):
        super().__init__(decoder)
        self.noiser = UnsupervisedMTNoising(
            dictionary=self.decoder.dictionary,
            max_word_shuffle_distance=3,
            word_dropout_prob=0.1,
            word_blanking_prob=0.1
        )
        self.interface = interface
        self.biling_dict = load_biling_dict(path=args.biling_dict_path, src_lang='de', tgt_lang='en')
        self.bpe = fastBPE.fastBPE(args.bpe_codes, args.bpe_vocab)

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

        interface = Embedding(len(task.source_dictionary), args.decoder_input_dim, padding_idx=task.source_dictionary.pad()) 
        interface.load_state_dict(torch.load(args.semface_path), strict=False)
        for m in interface.parameters():
            m.requires_grad_(False)

        return cls(decoder, interface, args)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, padding_idx=dictionary.pad())
        torch.nn.init.uniform_(embed_tokens.weight, a=-1e-4, b=1e-4) # SmallInit(Emb)
        torch.nn.init.constant_(embed_tokens.weight[dictionary.pad()], 0)
    
        if args.reload_xlit_encoder_emb:
            embed_tokens.load_state_dict(torch.load(args.semface_path), strict=False)
            for m in embed_tokens.parameters():
                m.requires_grad_(False)
                
        return embed_tokens
        
    def add_noise(self, src_tokens, src_lengths):
        """
        Add noise to the encoder input.
        """
        # word noising expect tensor of shape TxB
        # modified_x = self.create_code_switched_sentence(src_tokens, replace_prob=0.5)
        # modified_x, modified_lengths = self.noiser.noising(modified_x.t().cpu(), src_lengths.cpu())
        
        modified_x, modified_lengths = self.noiser.noising(src_tokens.t().cpu(), src_lengths.cpu())
        
        # Transpose back to expected src_tokens format
        modified_x = modified_x.t().to(src_tokens.device)
        modified_lengths = modified_lengths.to(src_tokens.device)
        
        return modified_x, modified_lengths
    
    def create_code_switched_sentence(self, src_tokens, replace_prob):
        dico = self.decoder.dictionary
        bsz, src_len = src_tokens.size()
        
        cs_tokens = torch.empty_like(src_tokens)
        # from IPython import embed; embed()
        for i in range(bsz):  
            sentence_str = remove_bpe(dico.string(src_tokens[i]))
            sentence = [s for s in sentence_str.split() if s != '<unk>' and s != '<pad>']

            replaced_sent, _ = replace_one_sent(sentence, self.biling_dict, replace_prob=replace_prob)
            replaced_sent = " ".join(replaced_sent)
            
            cs_sent = self.bpe.apply([replaced_sent])[0]
            while len(cs_sent.split()) < src_len:
                cs_sent = cs_sent + ' ' + dico.pad_word
            
            # cs tokens after bpe may exceed max src len
            # so we need to cap it off
            tmp_tokens = dico.encode_line(cs_sent, add_if_not_exist=False)
            cs_tokens[i] = tmp_tokens[:src_len]  
            
        return cs_tokens
    
    def forward(self, src_tokens, **kwargs):        
        # src_tokens: B x T
        
        # cs_tokens = self.create_code_switched_sentence(src_tokens)
        # cs_tokens_out = self.interface(cs_tokens)
        # encoder_out = self.interface(src_tokens)
        
        noisy_sample, _ = self.add_noise(src_tokens, kwargs['src_lengths'])
        encoder_out = self.interface(noisy_sample)

        # B x T x C -> T x B x C
        encoder_out = encoder_out.transpose(0, 1)
        # cs_tokens_out = cs_tokens_out.transpose(0, 1)
        # encoder_out = {
        #     "encoder_out": [encoder_out],  # T x B x C
        #     "encoder_padding_mask": [src_tokens.eq(self.decoder.padding_idx)],
        #     "cs_tokens": [cs_tokens_out],
        #     "cs_tokens_padding_mask": [cs_tokens.eq(self.decoder.padding_idx)],
        # }
        encoder_out = {
            "encoder_out": [encoder_out],  # T x B x C
            "encoder_padding_mask": [noisy_sample.eq(self.decoder.padding_idx)],
            "cs_tokens": [],
            "cs_tokens_padding_mask": [],
        }
        return self.decoder(src_tokens, encoder_out)


def remove_bpe(sent: str, bpe_symbol="@@ "):
    return (sent.replace("\n", '') + ' ').replace(bpe_symbol, '').rstrip()    
    
def load_biling_dict(path, src_lang, tgt_lang):
    """
    
    :param path: the path of the root of MUSE dictionaries
    :param languages: en-x dictionaries will be used for all x in `languages`
    :return: a dictionary of dictionaries that stores all word pairs for all en-x in `languages`
    """ 
    pair_name = f"{src_lang}-{tgt_lang}"
    
    if not os.path.isfile(os.path.join(path, pair_name+".txt")):
        raise FileNotFoundError("{}/{} not exists!".format(path, pair_name+".txt"))
    
    biling_dict = defaultdict(list)
    with open(os.path.join(path, pair_name+".txt")) as f:
        for _line in f:
            i = 0
            bi_text = _line.strip().split()
            assert len(bi_text) == 2, ("in file {}/{}, line index {} has an invalid number of columns {}"
                                        .format(path, pair_name, i, len(bi_text)))
            biling_dict[bi_text[0]].append(bi_text[1])
            i += 1

    return biling_dict

def replace_one_sent(sent: list, biling_dict: defaultdict, replace_prob=1.0):
    """

    :param sent: splitted source sentence (each token is string)
    :param biling_dict: is a default dict with list as key
    :return:
    """
    cnt = 0
    new_sent = []
    for i in range(len(sent)):
        if sent[i] in biling_dict and random.random() < replace_prob:
            new_sent.append(random.choice(biling_dict[sent[i]]))
            cnt += 1
        else:
            new_sent.append(sent[i])

    return new_sent, cnt

def load_multi_biling_dicts(path, langs):
    """
    
    :param path: the path of the root of MUSE dictionaries
    :param languages: en-x dictionaries will be used for all x in `languages`
    :return: a dictionary of dictionaries that stores all word pairs for all en-x in `languages`
    """
    biling_dict = {}
    pairs = list(permutations(langs, r=2))
    for (src_lang, tgt_lang) in pairs:
        pair_name = f"{src_lang}-{tgt_lang}"
        
        if not os.path.isfile(os.path.join(path, pair_name+".txt")):
            raise FileNotFoundError("{}/{} not exists!".format(path, pair_name+".txt"))
        
        x2y = defaultdict(list)
        with open(os.path.join(path, pair_name+".txt")) as f:
            for _line in f:
                i = 0
                bi_text = _line.strip().split()
                assert len(bi_text) == 2, ("in file {}/{}, line index {} has an invalid number of columns {}"
                                            .format(path, pair_name, i, len(bi_text)))
                x2y[bi_text[0]].append(bi_text[1])
                i += 1
        biling_dict[pair_name] = x2y

    return biling_dict

def xlit_architecture(args):
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
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 2048)
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

    args.use_rope = safe_getattr(
        args, "use_rope", False
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
    
    args.reload_xlit_encoder_emb = safe_getattr(args, "reload_xlit_encoder_emb", False)
    
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("xlit_decoder", "xlit_decoder_base")
def xlit_decoder(args):
    args.share_decoder_input_output_embed = safe_getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.decoder_layers = safe_getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 8)
    
    args.use_rope = safe_getattr(args, "use_rope", False)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)

    xlit_architecture(args)
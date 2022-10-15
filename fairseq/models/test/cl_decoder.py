# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
# 
# from typing import Dict, List, Optional, Tuple
# import os
# 
# 
# import torch
# import torch.nn as nn
# from torch import Tensor
# import numpy as np
# 
# from fairseq import utils
# from fairseq.dataclass.utils import gen_parser_from_dataclass
# from fairseq.distributed import fsdp_wrap
# from fairseq.models import FairseqEncoderDecoderModel
# from fairseq.models.transformer import (
#     TransformerConfig,
#     TransformerDecoderBase,
#     TransformerEncoderBase,
# )
# 
# 
# class CLDecoderBase(FairseqEncoderDecoderModel):
#     """
#     Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
#     <https://arxiv.org/abs/1706.03762>`_.
# 
#     Args:
#         encoder (TransformerEncoder): the encoder
#         decoder (TransformerDecoder): the decoder
# 
#     The Transformer model provides the following named architectures and
#     command-line arguments:
# 
#     .. argparse::
#         :ref: fairseq.models.transformer_parser
#         :prog:
#     """
#     embed_tokens = None
# 
#     def __init__(self, cfg, encoder, decoder):
#         super().__init__(encoder, decoder)
#         self.cfg = cfg
#         self.supported_targets = ["past", "self", "future"]
#         self.supports_align_args = True
# 
#     @classmethod
#     def add_args(cls, parser):
#         """Add model-specific arguments to the parser."""
#         # we want to build the args recursively in this case.
#         gen_parser_from_dataclass(
#             parser, TransformerConfig(), delete_default=False, with_prefix=""
#         )
# 
#     @classmethod
#     def build_model(cls, cfg, task):
#         """Build a new model instance."""
# 
#         # --  TODO T96535332
#         #  bug caused by interaction between OmegaConf II and argparsing
#         cfg.decoder.input_dim = int(cfg.decoder.input_dim)
#         cfg.decoder.output_dim = int(cfg.decoder.output_dim)
#         # --
# 
#         if cfg.encoder.layers_to_keep:
#             cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
#         if cfg.decoder.layers_to_keep:
#             cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))
# 
#         src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
#         CLDecoderBase.embed_tokens = cls.build_embedding(
#             cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
#         )
#         if cfg.share_all_embeddings:
#             if src_dict != tgt_dict:
#                 raise ValueError("--share-all-embeddings requires a joined dictionary")
#             if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
#                 raise ValueError(
#                     "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
#                 )
#             if cfg.decoder.embed_path and (
#                     cfg.decoder.embed_path != cfg.encoder.embed_path
#             ):
#                 raise ValueError(
#                     "--share-all-embeddings not compatible with --decoder-embed-path"
#                 )
#             cls.embed_tokens = cls.build_embedding(
#                 cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
#             )
#             decoder_embed_tokens = encoder_embed_tokens = cls.embed_tokens
#             cfg.share_decoder_input_output_embed = True
#         else:
#             encoder_embed_tokens = cls.build_embedding(
#                 cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
#             )
#             decoder_embed_tokens = cls.build_embedding(
#                 cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
#             )
#         if cfg.offload_activations:
#             cfg.checkpoint_activations = True  # offloading implies checkpointing
#         encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
#         decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
#         if not cfg.share_all_embeddings:
#             # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
#             encoder = fsdp_wrap(encoder, min_num_params=cfg.min_params_to_wrap)
#             decoder = fsdp_wrap(decoder, min_num_params=cfg.min_params_to_wrap)
#         return cls(cfg, encoder, decoder)
# 
#     @classmethod
#     def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
#         num_embeddings = len(dictionary)
#         # for i in range(20):
#         #     print(dictionary.__getitem__(i))
#         # assert False
#         padding_idx = dictionary.pad()
# 
#         emb = Embedding(num_embeddings, embed_dim, padding_idx)
#         # if provided, load from preloaded dictionaries
#         # if path:
#         #     embed_dict = utils.parse_embedding(path)
#         #     utils.load_embedding(embed_dict, dictionary, emb)
#         return emb
# 
#     @classmethod
#     def build_encoder(cls, cfg, src_dict, embed_tokens):
#         return TransformerEncoderBase(cfg, src_dict, embed_tokens)
# 
#     @classmethod
#     def build_decoder(cls, cfg, tgt_dict, embed_tokens):
#         return TransformerDecoderBase(
#             cfg,
#             tgt_dict,
#             embed_tokens,
#             no_encoder_attn=cfg.no_cross_attention,
#         )
# 
#     # TorchScript doesn't support optional arguments with variable length (**kwargs).
#     # Current workaround is to add union of all arguments in child classes.
#     def forward(
#             self,
#             src_tokens,
#             src_lengths,
#             masked_tokens=None,
#             prev_output_tokens=None,
#             return_all_hiddens: bool = True,
#             features_only: bool = False,
#             alignment_layer: Optional[int] = None,
#             alignment_heads: Optional[int] = None,
#     ):
#         """
#         Run the forward pass for an encoder-decoder model.
# 
#         Copied from the base class, but without ``**kwargs``,
#         which are not supported by TorchScript.
#         """
#         # from IPython import embed
#         # embed()
# 
#         # TODO: create noisy sample
#         # p_wd = 0.1  # word dropout probability
#         # mask_wd =  np.random.choice([True, False], size=src_lengths, p=[p_wd, 1-p_wd])
#         # src_tokens = src_tokens.masked_fill(torch.tensor(mask), self.decoder.dictionary.unk_index)
# 
#         def word_shuffle(src_tokens, src_lengths, shuffle_rate=3):
#             """
#             Randomly shuffle input words.
#             """
#             # define noise word scores
#             noise = np.random.uniform(0, shuffle_rate, size=(src_tokens.size(0) - 1, src_tokens.size(1)))
#             noise[-1] = -1  # do not move EOS symbol
# 
#             # be sure to shuffle entire words
#             bpe_end = self.bpe_end[lang_id][x]
#             word_idx = bpe_end[::-1].cumsum(0)[::-1]
#             word_idx = word_idx.max(0)[None, :] - word_idx
# 
#             assert shuffle_rate > 1
#             x2 = x.clone()
#             for i in range(l.size(0)):
#                 # generate a random permutation
#                 scores = word_idx[:l[i] - 1, i] + noise[word_idx[:l[i] - 1, i], i]
#                 scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
#                 permutation = scores.argsort()
#                 # shuffle words
#                 x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
#             return x2, l
# 
#         def word_dropout(src_tokens, src_lengths, dropout_rate=.1):
#             """
#             Randomly drop input words.
#             """
#             if self.params.word_dropout == 0:
#                 return x, l
#             assert 0 < self.params.word_dropout < 1
# 
#             # define words to drop
#             bos_index = self.params.bos_index[lang_id]
#             assert (x[0] == bos_index).sum() == l.size(0)
#             keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
#             keep[0] = 1  # do not drop the start sentence symbol
# 
#             # be sure to drop entire words
#             bpe_end = self.bpe_end[lang_id][x]
#             word_idx = bpe_end[::-1].cumsum(0)[::-1]
#             word_idx = word_idx.max(0)[None, :] - word_idx
# 
#             sentences = []
#             lengths = []
#             for i in range(l.size(0)):
#                 assert x[l[i] - 1, i] == self.params.eos_index
#                 words = x[:l[i] - 1, i].tolist()
#                 # randomly drop words from the input
#                 new_s = [w for j, w in enumerate(words) if keep[word_idx[j, i], i]]
#                 # we need to have at least one word in the sentence (more than the start / end sentence symbols)
#                 if len(new_s) == 1:
#                     new_s.append(words[np.random.randint(1, len(words))])
#                 new_s.append(self.params.eos_index)
#                 assert len(new_s) >= 3 and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
#                 sentences.append(new_s)
#                 lengths.append(len(new_s))
#             # re-construct input
#             l2 = torch.LongTensor(lengths)
#             x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
#             for i in range(l2.size(0)):
#                 x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
#             return x2, l2
# 
#         def word_blank(src_tokens, src_lengths):
#             """
#             Randomly blank input words.
#             """
#             if self.params.word_blank == 0:
#                 return x, l
#             assert 0 < self.params.word_blank < 1
# 
#             # define words to blank
#             bos_index = self.params.bos_index[lang_id]
#             assert (x[0] == bos_index).sum() == l.size(0)
#             keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
#             keep[0] = 1  # do not blank the start sentence symbol
# 
#             # be sure to blank entire words
#             bpe_end = self.bpe_end[lang_id][x]
#             word_idx = bpe_end[::-1].cumsum(0)[::-1]
#             word_idx = word_idx.max(0)[None, :] - word_idx
# 
#             sentences = []
#             for i in range(l.size(0)):
#                 assert x[l[i] - 1, i] == self.params.eos_index
#                 words = x[:l[i] - 1, i].tolist()
#                 # randomly blank words from the input
#                 new_s = [w if keep[word_idx[j, i], i] else self.params.blank_index for j, w in enumerate(words)]
#                 new_s.append(self.params.eos_index)
#                 assert len(new_s) == l[i] and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
#                 sentences.append(new_s)
#             # re-construct input
#             x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
#             for i in range(l.size(0)):
#                 x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
#             return x2, l
# 
#         def add_noise(self, words, lengths, lang_id):
#             """
#             Add noise to the encoder input.
#             """
#             words, lengths = self.word_shuffle(words, lengths, lang_id)
#             words, lengths = self.word_dropout(words, lengths, lang_id)
#             words, lengths = self.word_blank(words, lengths, lang_id)
#             return words, lengths
# 
#         # print(f"Vocab size: {self.encoder.dictionary.__len__()}")
#         # print(f"Padding: {self.encoder.padding_idx}")
#         # print(f"BOS: {self.encoder.dictionary.bos()}")
#         # print(f"EOS: {self.encoder.dictionary.eos()}")
#         #
#         # print("\n=====")
#         # print("Tokens:")
#         # print(src_tokens)
#         # print(src_tokens.size())
#         # print("=====\n")
#         # print("Before:")
#         # print(src_lengths)
#         # print(src_lengths.size())
#         # src_lengths = (
#         #     src_tokens.ne(self.encoder.padding_idx)
#         #     .sum(dim=1, dtype=torch.int32)
#         #     .reshape(-1, 1)
#         #     .contiguous()
#         # )
#         # print("After: ")
#         # print(src_lengths)
#         # print(src_lengths.size())
#         # print("=====\n")
# 
#         # project masked tokens only
#         if masked_tokens is not None:
#             src_tokens = src_tokens[masked_tokens, :]
#         encoder_padding_mask = src_tokens.eq(self.encoder.padding_idx)
#         has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
#         embed_tokens = CLDecoderBase.embed_tokens
# 
#         # bring to current device
#         embed_tokens.to(torch.device(f'cuda:{torch.cuda.current_device()}'))
# 
# 
#         token_embedding = embed_tokens(src_tokens)
#         x = encoder_embedding = self.encoder.embed_scale * token_embedding
#         if self.encoder.embed_positions is not None:
#             x = encoder_embedding + self.encoder.embed_positions(src_tokens)
#         if self.encoder.layernorm_embedding is not None:
#             x = self.encoder.layernorm_embedding(x)
#         x = self.encoder.dropout_module(x)
#         # account for padding while computing the representation
#         if has_pads:
#             x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
#         # B x T x C -> T x B x C
#         x = x.transpose(0, 1)
# 
#         encoder_states = []
#         fc_results = []
# 
#         encoder_out = {
#             "encoder_out": [x],  # T x B x C
#             "encoder_padding_mask": [encoder_padding_mask],  # B x T
#             "encoder_embedding": [encoder_embedding],  # B x T x C
#             "encoder_states": encoder_states,  # List[T x B x C]
#             "fc_results": fc_results,  # List[T x B x C]
#             "src_tokens": [],
#             "src_lengths": [src_lengths],
#         }
# 
#         decoder_out = self.decoder(
#             prev_output_tokens,
#             encoder_out=encoder_out,
#             features_only=features_only,
#             alignment_layer=alignment_layer,
#             alignment_heads=alignment_heads,
#             src_lengths=src_lengths,
#             return_all_hiddens=return_all_hiddens,
#         )
#         return decoder_out
# 
#     # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
#     # I rewrite the get_normalized_probs from Base Class to call the
#     # helper function in the Base Class.
#     @torch.jit.export
#     def get_normalized_probs(
#             self,
#             net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
#             log_probs: bool,
#             sample: Optional[Dict[str, Tensor]] = None,
#     ):
#         """Get normalized probabilities (or log probs) from a net's output."""
#         return self.get_normalized_probs_scriptable(net_output, log_probs, sample)
# 
# 
# # from fairseq.models.masked_lm import MaskedLMModel
# 
# # model = MaskedLMModel.from_pretrained(
# #   '/content/checkpoints/mlm',
# #   checkpoint_file="checkpoint_best.pt",
# #   data_name_or_path="/content/data/iwslt14/fairseq_processed",
# # )
# # pretrained_emb = model.get_submodule("models.0.encoder.sentence_encoder.embed_tokens")
# 
# 
# def Embedding(num_embeddings, embedding_dim, padding_idx):
#     # TODO: change path
#     pretrained_emb = torch.nn.Embedding(num_embeddings,embedding_dim, padding_idx=padding_idx)
#     pretrained_emb.load_state_dict(torch.load('/root/khang/data-bin/trained_embed.pt'))
#     return pretrained_emb

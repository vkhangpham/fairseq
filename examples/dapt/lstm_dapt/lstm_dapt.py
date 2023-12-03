import logging, sys
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.lstm import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    LSTMModel,
    LSTMEncoder,
    LSTMDecoder,
    AttentionLayer,
    LSTM,
    LSTMCell,
    Linear,
    base_architecture,
)
from torch import Tensor


logger = logging.getLogger(__name__)

@register_model("lstm_dapt")
class LSTMDAPTModel(LSTMModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
    
    @staticmethod
    def add_args(parser):
        LSTMModel.add_args(parser)
        parser.add_argument('--source-position-markers', type=int, metavar='N',
                            help='dictionary includes N additional items that '
                                 'represent an OOV token at a particular input '
                                 'position')
        parser.add_argument('--use-coverage', default=False, action='store_true',
                            help="use coverage mechanism")

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        if getattr(args, "source_position_markers", None) is None:
            args.source_position_markers = args.max_source_positions

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if src_dict != tgt_dict:
            raise ValueError("Pointer-generator requires a joined dictionary")
        
        if args.encoder_layers != args.decoder_layers:
            raise ValueError("--encoder-layers must match --decoder-layers")

        max_source_positions = getattr(
            args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
        )
        max_target_positions = getattr(
            args, "max_target_positions", DEFAULT_MAX_TARGET_POSITIONS
        )

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            embed_tokens = build_embedding(dictionary, args.source_position_markers, embed_dim)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim
            )
        else:
            pretrained_encoder_embed = build_embedding(
                src_dict, args.source_position_markers, args.encoder_embed_dim
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError("--share-all-embeddings requires a joint dictionary")
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embed not compatible with --decoder-embed-path"
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to "
                    "match --decoder-embed-dim"
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim,
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
            args.decoder_embed_dim != args.decoder_out_embed_dim
        ):
            raise ValueError(
                "--share-decoder-input-output-embeddings requires "
                "--decoder-embed-dim to match --decoder-out-embed-dim"
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = LSTMDAPTEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions,
            source_position_markers=args.source_position_markers
        )
        decoder = LSTMDAPTDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=utils.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            max_target_positions=max_target_positions,
            residuals=False,
            source_position_markers=args.source_position_markers,
            use_coverage=args.use_coverage
        )
        return cls(encoder, decoder)
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )
        return decoder_out

class LSTMDAPTEncoder(LSTMEncoder):
    def __init__(
        self, 
        dictionary, 
        embed_dim=512, 
        hidden_size=512, 
        num_layers=1, 
        dropout_in=0.1, 
        dropout_out=0.1, 
        bidirectional=False, 
        left_pad=True, 
        pretrained_embed=None, 
        padding_idx=None, 
        max_source_positions=...,
        source_position_markers=1000
    ):
        super().__init__(
            dictionary, 
            embed_dim, 
            hidden_size, 
            num_layers, 
            dropout_in, 
            dropout_out, 
            bidirectional, 
            left_pad, 
            pretrained_embed, 
            padding_idx, 
            max_source_positions
        )
        if pretrained_embed is None:
            self.embed_tokens = build_embedding(dictionary, source_position_markers, embed_dim)
        else:
            self.embed_tokens = pretrained_embed
        self.self_attn = LSTMSelfAttention(hidden_size=hidden_size)

    def forward(self, src_tokens: Tensor, src_lengths: Tensor, enforce_sorted: bool = True):
        x, final_hiddens, final_cells, encoder_padding_mask = super().forward(src_tokens, src_lengths, enforce_sorted)
        z, _ = self.self_attn(x, encoder_padding_mask)
        return tuple(
            (   
                x,
                final_hiddens,
                final_cells,
                encoder_padding_mask,
                z,  # key information vector
                src_tokens,
            )
        )
    def reorder_encoder_out(
        self, encoder_out: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], new_order
    ):
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
                encoder_out[3].index_select(1, new_order),
                encoder_out[4].index_select(0,new_order),
                encoder_out[5].index_select(0,new_order),
            )
        )
      
class LSTMDAPTDecoder(LSTMDecoder):
    def __init__(
        self, 
        dictionary, 
        embed_dim=512, 
        hidden_size=512, 
        out_embed_dim=512, 
        num_layers=1, 
        dropout_in=0.1, 
        dropout_out=0.1, 
        attention=True, 
        encoder_output_units=512, 
        pretrained_embed=None, 
        share_input_output_embed=False, 
        adaptive_softmax_cutoff=None, 
        max_target_positions=..., 
        residuals=False,
        source_position_markers=1000,
        use_coverage=False
    ):
        super().__init__(
            dictionary, 
            embed_dim, 
            hidden_size, 
            out_embed_dim, 
            num_layers, 
            dropout_in, 
            dropout_out, 
            attention, 
            encoder_output_units, 
            pretrained_embed, 
            share_input_output_embed, 
            adaptive_softmax_cutoff, 
            max_target_positions, 
            residuals
        )
        self.attention = LSTMSoftAttention(self.hidden_size, use_coverage=use_coverage)
        self.num_types = len(dictionary)
        self.num_oov_types = source_position_markers
        self.num_embeddings = self.num_types - self.num_oov_types

        if pretrained_embed is None:
            self.embed_tokens = build_embedding(dictionary, source_position_markers, embed_dim)
        else:
            self.embed_tokens = pretrained_embed

        if not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, self.num_embeddings, dropout=dropout_out)

        # gate mechanism
        self.x_context = Linear(2 * hidden_size + embed_dim, 2 * embed_dim)
        self.gm_projection = Linear(4 * hidden_size + embed_dim, 1)

        # pointer generation
        self.pgen_projection = Linear(4 * hidden_size + embed_dim, 1)  # state size: 2*hidden, context_vec: 2*hidden

        self.out_projection = Linear(2*hidden_size, hidden_size)

        self.coverage = None
        self.use_coverage = use_coverage
    
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.
        """
        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out[0]
            encoder_hiddens = encoder_out[1]
            encoder_cells = encoder_out[2]
            encoder_padding_mask = encoder_out[3]
        else:
            encoder_outs = torch.empty(0)
            encoder_hiddens = torch.empty(0)
            encoder_cells = torch.empty(0)
            encoder_padding_mask = torch.empty(0)
        srclen = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(
                incremental_state
            )
        elif encoder_out is not None:
            # setup recurrent cells
            prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(y) for y in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size*2)  # bc context vec size is 2*hidden
        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = None

        assert (
            srclen > 0 or self.attention is None
        ), "attention is not supported if there are no encoder outputs"
        attn_scores: Optional[Tensor] = (
            x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        )

        key_info = encoder_out[4]  # B x N
        outs = []
        pgen_list = []
        cov_list = []

        for j in range(seqlen):
            coverage = None
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)  # B x C + 2*C
                input = self.x_context(input)  # B x 2*C
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                assert attn_scores is not None
                out, attn_scores[:, j, :], coverage = self.attention(
                    hidden, encoder_outs, encoder_padding_mask, coverage
                )  #  out: B x N
                cov_list.append(coverage.unsqueeze(1))  # T1 x 1 x B
            else:
                out = hidden

            # input feeding
            if input_feed is not None:
                input_feed = out

            state = torch.cat(
                [
                    hidden.reshape(-1, self.hidden_size),
                    cell.reshape(-1, self.hidden_size)
                ],
                dim=1
            )  # B x N

            # dec_input = self.x_context(torch.cat([out,x[j]], dim=-1))  # B x C
            gate_input = torch.cat([key_info, state, input], dim=-1)  # B x 2*C + 2*C + C
            gate_input = self.gm_projection(gate_input)   # B x 1
            gate = F.sigmoid(gate_input)  # B

            mixture = (1-gate)*out + gate*key_info  # B x 2*C
            p_gen_input = torch.cat([mixture, state, input], dim=-1)
            p_gen = self.pgen_projection(p_gen_input)  # B x 1
            pgen_list.append(p_gen)

            out = self.out_projection(mixture)
            out = self.dropout_out_module(out)

            # save final output
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)
        p_gens = torch.cat(pgen_list, dim=-1).unsqueeze(-1)  # B x T2 x 1
        coverages = torch.cat(cov_list, dim=1)  # T1 x T2 x B
        coverages = coverages.transpose(0,2)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, "additional_fc") and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)
        return x, attn_scores, p_gens, coverages

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        src_lengths: Optional[Tensor] = None,
    ):
        x, attn_scores, p_gens, coverage = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )

        return self.output_layer(x, attn_scores, p_gens, encoder_out[5]), attn_scores, coverage

    def output_layer(self, x, attn_scores, p_gens, src_tokens):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)

        # x: B x T x V
        batch_size, output_length, vocab_size = x.shape
        src_length = src_tokens.shape[1]

        vocab_dist = torch.mul(x, p_gens)
        padding_size = (batch_size, output_length, self.num_oov_types)
        padding = vocab_dist.new_zeros(padding_size)
        vocab_dist = torch.cat((vocab_dist, padding), 2)

        attn = torch.mul(attn_scores, 1.0 - p_gens)
        index = src_tokens[:, None, :]
        index = index.expand(batch_size, output_length, src_length)
        attn_dists_size = (batch_size, output_length, self.num_types)
        attn_dists = attn.new_zeros(attn_dists_size)
        attn_dists.scatter_add_(2, index, attn)

        # Final distributions, [batch_size, output_length, num_types].
        return vocab_dist + attn_dists
    

class LSTMSoftAttention(nn.Module):
    def __init__(self, hidden_size, use_coverage=False, cov_truncation=None):
        super().__init__()
        self.use_coverage = use_coverage
        self.cov_truncation = cov_truncation
        if use_coverage:
            self.coverage_projection = Linear(1, hidden_size * 2, bias=False)
        self.decode_proj = Linear(hidden_size, hidden_size * 2)
        self.v = Linear(hidden_size * 2, 1, bias=False)


    def forward(self, decoder_hidden_states, encoder_outs, encoder_padding_mask, coverage):
        T, B, N = encoder_outs.shape
        if coverage is None:
            coverage = torch.zeros([T,B], dtype=encoder_outs.dtype, device=encoder_outs.device)

        dec_fea = self.decode_proj(decoder_hidden_states)  # B x N
        dec_fea_expanded = dec_fea.unsqueeze(0).expand(T, B, N)  # T x B x N
        # dec_fea_expanded = torch.reshape(dec_fea_expanded, [-1, N])  # T*B x N

        att_features = encoder_outs + dec_fea_expanded  # T x B x N
        if self.use_coverage:
            coverage_input = torch.reshape(coverage, [-1, 1])  # T*B x 1
            coverage_feature = self.coverage_projection(coverage_input)  # T*B x N
            coverage_feature = coverage_feature.reshape(att_features.shape)
            att_features = att_features + coverage_feature

        e = F.tanh(att_features)  # T x B x N
        attn_scores = self.v(e)  # T x B x 1
        attn_scores = torch.reshape(attn_scores, [T, -1])  # T x B

        if encoder_padding_mask is not None:
            attn_scores = (
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )

        attn_dist = F.softmax(attn_scores, dim=0)  # T x B
        c = (attn_dist.unsqueeze(2) * encoder_outs).sum(dim=0)  # B x N

        # if self.use_coverage:
        #     coverage = torch.where(
        #         attn_dist > self.cov_truncation,
        #         coverage + attn_dist,
        #         coverage + sys.float_info.epsilon,
        #     )
        if self.use_coverage:
            coverage = coverage + attn_dist
        return c, attn_dist, coverage
class LSTMSelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = Linear(2*hidden_size, 2*hidden_size)
        self.v = Linear(2*hidden_size, 1, bias=False)

    def forward(self, encoder_outs, encoder_padding_mask):
        T, B, N = encoder_outs.shape

        attn_features = self.W(encoder_outs)  # T x B x N
        attn_features = encoder_outs + attn_features  # T x B x N
        attn_scores = self.v(F.tanh(attn_features))  # T x B x 1
        attn_scores = attn_scores.reshape(T, -1)  # T x B

        if encoder_padding_mask is not None:
            attn_scores = (
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )

        attn_dist = F.softmax(attn_scores, dim=0)  # T x B
        z = (attn_dist.unsqueeze(-1) * encoder_outs).sum(dim=0)  # B x N

        return z, attn_dist
    
def build_embedding(dictionary, source_position_markers, embed_dim, path=None):
    # The dictionary may include additional items that can be used in
    # place of the normal OOV token and that all map to the same
    # embedding. Using a different token for each input position allows
    # one to restore the word identities from the original source text.
    num_embeddings = len(dictionary) - source_position_markers
    padding_idx = dictionary.pad()
    unk_idx = dictionary.unk()
    logger.info(
        "dictionary indices from {0} to {1} will be mapped to {2}".format(
            num_embeddings, len(dictionary) - 1, unk_idx
        )
    )
    emb = Embedding(num_embeddings, embed_dim, padding_idx, unk_idx)
    # if provided, load from preloaded dictionaries
    if path:
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, emb)
    return emb

class Embedding(nn.Embedding):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings. This subclass differs from the standard PyTorch Embedding class by
    allowing additional vocabulary entries that will be mapped to the unknown token
    embedding.
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int): Pads the output with the embedding vector at :attr:`padding_idx`
                           (initialized to zeros) whenever it encounters the index.
        unk_idx (int): Maps all token indices that are greater than or equal to
                       num_embeddings to this index.
    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`
    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)
    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.
    """
    __constants__ = ["unk_idx"]

    # Torchscript: Inheriting from Embedding class produces an error when exporting to Torchscript
    # -> RuntimeError: Unable to cast Python instance to C++ type (compile in debug mode for details
    # It's happening because max_norm attribute from nn.Embedding is None by default and it cannot be
    # cast to a C++ type
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int],
        unk_idx: int,
        max_norm: Optional[float] = float("inf"),
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm)
        self.unk_idx = unk_idx
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(self.weight[padding_idx], 0)

    def forward(self, input):
        input = torch.where(
            input >= self.num_embeddings, torch.ones_like(input) * self.unk_idx, input
        )
        return nn.functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )

    
@register_model_architecture("lstm_dapt", "dapt_base")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_hidden_size = getattr(
        args, "encoder_hidden_size", args.encoder_embed_dim
    )
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_bidirectional = getattr(args, "encoder_bidirectional", True)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_attention = getattr(args, "decoder_attention", "1")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None
    )

@register_model_architecture("lstm_dapt", "dapt_dummy")
def lstm_dummy(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 8)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_hidden_size = getattr(
        args, "encoder_hidden_size", args.encoder_embed_dim
    )
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_bidirectional = getattr(args, "encoder_bidirectional", True)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 8)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 8)
    args.decoder_attention = getattr(args, "decoder_attention", "1")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,50000,200000"
    )

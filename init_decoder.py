from fairseq.models.masked_lm import MaskedLMModel
from fairseq.models.cl_decoder import CLDecoder
import torch
import os, sys

args = sys.argv
assert len(args) == 5

enc_path = args[1]
enc_data_path = args[2]
dec_path = args[3]
dec_data_path = args[4]

enc_model = MaskedLMModel.from_pretrained(
    enc_path,
    "checkpoint_best.pt",
    enc_data_path
)

dec_model = CLDecoder.from_pretrained(
    dec_path,
    "checkpoint_best.pt",
    dec_data_path
)

encoder = enc_model.get_submodule('models.0.encoder.sentence_encoder')
decoder = dec_model.get_submodule('models.0.decoder')

x = encoder.weight.data
x = torch.cat((x[:4,:], x[5:, :]))
decoder.output_projection.weight.data = torch.nn.Parameter(x)
os.path.join
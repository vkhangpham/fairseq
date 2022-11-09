from fairseq.models.masked_lm import MaskedLMModel
from fairseq.models.cl_decoder import CLDecoder
from fairseq.models.transformer import TransformerModel

import torch
from torch.nn import Embedding, Parameter

encoder_model = MaskedLMModel.from_pretrained(
    "checkpoints/80k/enc",
    "checkpoint_best.pt",
    "data-bin/80k"
)
encoder = encoder_model.get_submodule('models.0.encoder.sentence_encoder')
x = encoder.embed_tokens.weight.data
x = torch.cat((x[:4,:], x[5:,:]))
# encoder.embed_tokens = Embedding(x.size(0), x.size(1), padding_idx=1)
# encoder.embed_tokens.weight.data = Parameter(x)

decoder_model = CLDecoder.from_pretrained(
    "checkpoints/80k/dec",
    "checkpoint_best.pt",
    "data-bin/lm_80k")
decoder = decoder_model.get_submodule("models.0.decoder")

de2en = TransformerModel.from_pretrained(
    'checkpoints/placeholder',
    'checkpoint_best.pt',
    'data-bin/iwslt14_80k'
)

de2en.models[0].encoder.dropout_module = encoder.dropout_module
de2en.models[0].encoder.embed_tokens = Embedding(x.size(0), x.size(1), padding_idx=1)
de2en.models[0].encoder.embed_tokens.weight.data = Parameter(x)
for i in range(6):
    de2en.models[0].encoder.layers[i].self_attn = encoder.layers[i].self_attn
    de2en.models[0].encoder.layers[i].self_attn_layer_norm = encoder.layers[i].self_attn_layer_norm
    de2en.models[0].encoder.layers[i].dropout_module = encoder.layers[i].dropout_module
    de2en.models[0].encoder.layers[i].activation_dropout_module = encoder.layers[i].activation_dropout_module
    de2en.models[0].encoder.layers[i].fc1 = encoder.layers[i].fc1
    de2en.models[0].encoder.layers[i].fc2 = encoder.layers[i].fc2
    de2en.models[0].encoder.layers[i].final_layer_norm = encoder.layers[i].final_layer_norm
de2en.models[0].decoder = decoder

checkpoint = torch.load("checkpoints/placeholder/checkpoint_best.pt")
for key in checkpoint['model'].keys():
    checkpoint['model'][key] = de2en.state_dict()["models.0." + key]
torch.save(checkpoint, "checkpoints/80k/init/finetune.pt")
print("Done")
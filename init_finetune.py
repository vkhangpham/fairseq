from fairseq.models.masked_lm import MaskedLMModel
from fairseq.models.cl_decoder import CLDecoder
from fairseq.models.transformer import TransformerModel

import torch
from torch.nn import Parameter

encoder_model = MaskedLMModel.from_pretrained(
    "checkpoints/fr2en/encoder10M",
    "checkpoint_best.pt",
    "data/para/fr2en/bin/encoder10M"
)
encoder = encoder_model.get_submodule('models.0.encoder.sentence_encoder')
x = encoder.embed_tokens.weight.data

decoder_model = CLDecoder.from_pretrained(
    "checkpoints/fr2en/decoder10M",
    "checkpoint_best.pt",
    "data/para/fr2en/bin/decoder10M"
)
decoder = decoder_model.get_submodule("models.0.decoder")

translation_model = TransformerModel.from_pretrained(
    'checkpoints/fr2en/baseline',
    'checkpoint_best.pt',
    'data/para/fr2en/bin/baseline'
)

translation_model.models[0].encoder.dropout_module = encoder.dropout_module
translation_model.models[0].encoder.embed_tokens.weight.data = Parameter(encoder.embed_tokens.weight)

for i in range(len(translation_model.models[0].encoder.layers)):
    translation_model.models[0].encoder.layers[i].self_attn = encoder.layers[i].self_attn
    translation_model.models[0].encoder.layers[i].self_attn_layer_norm = encoder.layers[i].self_attn_layer_norm
    translation_model.models[0].encoder.layers[i].dropout_module = encoder.layers[i].dropout_module
    translation_model.models[0].encoder.layers[i].activation_dropout_module = encoder.layers[i].activation_dropout_module
    translation_model.models[0].encoder.layers[i].fc1 = encoder.layers[i].fc1
    translation_model.models[0].encoder.layers[i].fc2 = encoder.layers[i].fc2
    translation_model.models[0].encoder.layers[i].final_layer_norm = encoder.layers[i].final_layer_norm

# for i in range(len(translation_model.models[0].encoder.layers)):
#     translation_model.models[0].encoder.layers[i].self_attn.dropout_module = encoder.layers[i].self_attn.dropout_module
#     translation_model.models[0].encoder.layers[i].self_attn.q_proj.weight = Parameter(encoder.layers[i].self_attn.q_proj.weight.data)
#     translation_model.models[0].encoder.layers[i].self_attn.q_proj.bias = Parameter(encoder.layers[i].self_attn.q_proj.bias.data)
#     translation_model.models[0].encoder.layers[i].self_attn.k_proj.weight = Parameter(encoder.layers[i].self_attn.k_proj.weight.data)
#     translation_model.models[0].encoder.layers[i].self_attn.k_proj.bias = Parameter(encoder.layers[i].self_attn.k_proj.bias.data)
#     translation_model.models[0].encoder.layers[i].self_attn.v_proj.weight = Parameter(encoder.layers[i].self_attn.v_proj.weight.data)
#     translation_model.models[0].encoder.layers[i].self_attn.v_proj.bias = Parameter(encoder.layers[i].self_attn.v_proj.bias.data)
#     translation_model.models[0].encoder.layers[i].self_attn.out_proj.weight = Parameter(encoder.layers[i].self_attn.out_proj.weight.data)
#     translation_model.models[0].encoder.layers[i].self_attn.out_proj.bias = Parameter(encoder.layers[i].self_attn.out_proj.bias.data)
#     translation_model.models[0].encoder.layers[i].activation_dropout_module = encoder.layers[i].activation_dropout_module
#     translation_model.models[0].encoder.layers[i].self_attn_layer_norm.weight = Parameter(encoder.layers[i].self_attn_layer_norm.weight.data)
#     translation_model.models[0].encoder.layers[i].self_attn_layer_norm.bias = Parameter(encoder.layers[i].self_attn_layer_norm.bias.data)
#     translation_model.models[0].encoder.layers[i].fc1.weight = Parameter(encoder.layers[i].fc1.weight.data)
#     translation_model.models[0].encoder.layers[i].fc1.bias = Parameter(encoder.layers[i].fc1.bias.data)
#     translation_model.models[0].encoder.layers[i].fc2.weight = Parameter(encoder.layers[i].fc2.weight.data)
#     translation_model.models[0].encoder.layers[i].fc2.bias = Parameter(encoder.layers[i].fc2.bias.data)
#     translation_model.models[0].encoder.layers[i].dropout_module = encoder.layers[i].dropout_module
#     translation_model.models[0].encoder.layers[i].final_layer_norm.weight = Parameter(encoder.layers[i].final_layer_norm.weight.data)
#     translation_model.models[0].encoder.layers[i].final_layer_norm.bias = Parameter(encoder.layers[i].final_layer_norm.bias.data)

translation_model.models[0].decoder = decoder
# translation_model.models[0].decoder.output_projection.weight = Parameter(decoder.output_projection.weight.data)
# for i in range(len(translation_model.models[0].decoder.layers)):
#     translation_model.models[0].decoder.layers[i].dropout_module = decoder.layers[i].dropout_module
#     translation_model.models[0].decoder.layers[i].self_attn.dropout_module = decoder.layers[i].self_attn.dropout_module
#     translation_model.models[0].decoder.layers[i].self_attn.q_proj.weight = Parameter(decoder.layers[i].self_attn.q_proj.weight.data)
#     translation_model.models[0].decoder.layers[i].self_attn.q_proj.bias = Parameter(decoder.layers[i].self_attn.q_proj.bias.data)
#     translation_model.models[0].decoder.layers[i].self_attn.k_proj.weight = Parameter(decoder.layers[i].self_attn.k_proj.weight.data)
#     translation_model.models[0].decoder.layers[i].self_attn.k_proj.bias = Parameter(decoder.layers[i].self_attn.k_proj.bias.data)
#     translation_model.models[0].decoder.layers[i].self_attn.v_proj.weight = Parameter(decoder.layers[i].self_attn.v_proj.weight.data)
#     translation_model.models[0].decoder.layers[i].self_attn.v_proj.bias = Parameter(decoder.layers[i].self_attn.v_proj.bias.data)
#     translation_model.models[0].decoder.layers[i].self_attn.out_proj.weight = Parameter(decoder.layers[i].self_attn.out_proj.weight.data)
#     translation_model.models[0].decoder.layers[i].self_attn.out_proj.bias = Parameter(decoder.layers[i].self_attn.out_proj.bias.data)
#     translation_model.models[0].decoder.layers[i].activation_dropout_module = decoder.layers[i].activation_dropout_module
#     translation_model.models[0].decoder.layers[i].self_attn_layer_norm.weight = Parameter(decoder.layers[i].self_attn_layer_norm.weight.data)
#     translation_model.models[0].decoder.layers[i].self_attn_layer_norm.bias = Parameter(decoder.layers[i].self_attn_layer_norm.bias.data)
#     translation_model.models[0].decoder.layers[i].encoder_attn.dropout_module = decoder.layers[i].encoder_attn.dropout_module
#     translation_model.models[0].decoder.layers[i].encoder_attn.q_proj.weight = Parameter(decoder.layers[i].encoder_attn.q_proj.weight.data)
#     translation_model.models[0].decoder.layers[i].encoder_attn.q_proj.bias = Parameter(decoder.layers[i].encoder_attn.q_proj.bias.data)
#     translation_model.models[0].decoder.layers[i].encoder_attn.k_proj.weight = Parameter(decoder.layers[i].encoder_attn.k_proj.weight.data)
#     translation_model.models[0].decoder.layers[i].encoder_attn.k_proj.bias = Parameter(decoder.layers[i].encoder_attn.k_proj.bias.data)
#     translation_model.models[0].decoder.layers[i].encoder_attn.v_proj.weight = Parameter(decoder.layers[i].encoder_attn.v_proj.weight.data)
#     translation_model.models[0].decoder.layers[i].encoder_attn.v_proj.bias = Parameter(decoder.layers[i].encoder_attn.v_proj.bias.data)
#     translation_model.models[0].decoder.layers[i].encoder_attn.out_proj.weight = Parameter(decoder.layers[i].encoder_attn.out_proj.weight.data)
#     translation_model.models[0].decoder.layers[i].encoder_attn.out_proj.bias = Parameter(decoder.layers[i].encoder_attn.out_proj.bias.data)
#     translation_model.models[0].decoder.layers[i].encoder_attn_layer_norm.weight = Parameter(decoder.layers[i].encoder_attn_layer_norm.weight.data)
#     translation_model.models[0].decoder.layers[i].encoder_attn_layer_norm.bias = Parameter(decoder.layers[i].encoder_attn_layer_norm.bias.data)
#     translation_model.models[0].decoder.layers[i].fc1.weight = Parameter(decoder.layers[i].fc1.weight.data)
#     translation_model.models[0].decoder.layers[i].fc1.bias = Parameter(decoder.layers[i].fc1.bias.data)
#     translation_model.models[0].decoder.layers[i].fc2.weight = Parameter(decoder.layers[i].fc2.weight.data)
#     translation_model.models[0].decoder.layers[i].fc2.bias = Parameter(decoder.layers[i].fc2.bias.data)
#     translation_model.models[0].decoder.layers[i].final_layer_norm.weight = Parameter(decoder.layers[i].final_layer_norm.weight.data)
#     translation_model.models[0].decoder.layers[i].final_layer_norm.bias = Parameter(decoder.layers[i].final_layer_norm.bias.data)

checkpoint = torch.load("checkpoints/fr2en/baseline/checkpoint_best.pt")
for key in checkpoint['model'].keys():
    checkpoint['model'][key] = translation_model.state_dict()["models.0." + key]
torch.save(checkpoint, "checkpoints/fr2en/finetune10M/init.pt")
print("Done")

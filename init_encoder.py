from fairseq.models.masked_lm import MaskedLMModel
import torch

vm = {}
with open("../vecmap/result/joined.vec", 'r') as f:
    for line in f:
        s = line.split()
        vm[s[0]] = list(map(float, s[1:]))

enc_model = MaskedLMModel.from_pretrained(
    "checkpoints/80k/toy",
    "checkpoint_best.pt",
    "data-bin/toy/xlm"
)
fm = enc_model.get_submodule('models.0.encoder.sentence_encoder').dictionary.symbols

ind = []
for symbol in fm:
    if symbol in vm:
        ind.append((symbol, fm.index(symbol)))


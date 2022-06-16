# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data import Dictionary


class MaskedLMDictionary(Dictionary):
    """
    Dictionary for Masked Language Modelling tasks. This extends Dictionary by
    adding the mask symbol.
    """

    def __init__(
        self,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        mask="<mask>",
    ):
        super().__init__(pad=pad, eos=eos, unk=unk)
        extra_special_symbols = ['<special1>', '<special2>','<special3>','<special4>','<special5>','<special6>','<special7>','<special8>']
        for s in extra_special_symbols:
            self.add_symbol(s)
        self.mask_word = mask
        self.mask_index = self.add_symbol(mask)
        self.nspecial = len(self.symbols)

    def mask(self):
        """Helper to get index of mask symbol"""
        return self.mask_index


class BertDictionary(MaskedLMDictionary):
    """
    Dictionary for BERT task. This extends MaskedLMDictionary by adding support
    for cls and sep symbols.
    """

    def __init__(
        self,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        mask="<mask>",
        cls="<cls>",
        sep="<sep>",
    ):
        super().__init__(pad=pad, eos=eos, unk=unk, mask=mask)
        self.cls_word = cls
        self.sep_word = sep
        self.cls_index = self.add_symbol(cls)
        self.sep_index = self.add_symbol(sep)
        self.nspecial = len(self.symbols)

    def cls(self):
        """Helper to get index of cls symbol"""
        return self.cls_index

    def sep(self):
        """Helper to get index of sep symbol"""
        return self.sep_index

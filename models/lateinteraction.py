import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from transformers.modeling_outputs import BaseModelOutput
from models import InBatch

@dataclass
class LateInteractionOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    acc: Optional[Tuple[torch.FloatTensor, ...]] = None
    scores: torch.FloatTensor = None
    acc_m: Optional[Tuple[torch.FloatTensor, ...]] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class LateInteraction(InBatch):
    def __init__(self, opt, retriever, tokenizer, label_smoothing=False,):
        super(InBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = label_smoothing
        self.tokenizer = tokenizer
        self.encoder = retriever
        assert self.encoder.use_multivectors, 'the encoder should return multiple vetors instead of single'

        self.tau = opt.temperature
        self.skiplist = {w: True for symbol in string.punctuation for w in [symbol, tokenizer.encode(symbol, add_special_tokens=False)[0]]} 

    def mask(self, tokens):
        # exclude trival tokens
        mask = [[(x != self.tokenizer.pad_token_id) and (x not in self.skiplist) for x in d] for d in tokens.cpu().tolist()]
        # True means token to be considered
        return mask

    def forward(self, q_tokens, q_mask, c_tokens, c_mask, **kwargs):

        # this is for random cropping
        if (c_tokens is None) and (c_mask is None):
            return self.forward_bidirectional(q_tokens, q_mask, **kwargs)

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        CELoss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # multi-vector: (B L H)
        qembs = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        cembs = self.encoder(input_ids=c_tokens, attention_mask=c_mask, normalize=self.norm_doc)

        ## [Inbatch-s]: dotproduct
        scores_s = torch.einsum("id, jd->ij", qembs[:, 0] / self.tau, cembs[:, 0])
        loss_s = CELoss(scores_s, labels)

        predicted_idx = torch.argmax(scores_s, dim=-1)
        accuracy_s = 100 * (predicted_idx == labels).float().mean()

        ## [Inbatch-m]: maxsim
        # multiplcation: (1 B Lc H) x (B 1 H Lq) = (B B Lc Lq) --> (BB Lc Lq)
        scores_m = (cembs.unsqueeze(0) @ qembs.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)
        cmask = torch.tensor(self.mask(c_tokens), device=c_tokens.device).bool()
        scores_m[~cmask.repeat(bsz, 1)] = -9999
        scores_m = scores_m.max(1).values.sum(-1) # 
        scores_m = scores_m.view(bsz, -1)
        loss_m = CELoss(scores_m, labels)

        predicted_idx = torch.argmax(scores_m, dim=-1)
        accuracy_m = 100 * (predicted_idx == labels).float().mean()

        loss = loss_s + loss_m

        return LateInteractionOutput(
	        loss=loss, 
	        acc=accuracy_s, 
	        scores=scores_s, 
                logs={'loss_s': loss_s, 'loss_m': loss_m, 'acc_m': accuracy_m},
        )

    def get_encoder(self):
        return self.encoder


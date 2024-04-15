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
    logs: Optional[Dict[str, torch.FloatTensor]] = None
    scores: torch.FloatTensor = None
    acc_m: Optional[Tuple[torch.FloatTensor, ...]] = None

class LateInteraction(InBatch):
    def __init__(self, opt, retriever, tokenizer, label_smoothing=False, exclude_cls_token=False):
        super(InBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = label_smoothing
        self.tokenizer = tokenizer
        self.encoder = retriever

        self.tau = opt.temperature
        self.tau_p = opt.temperature_span
        self.tau_m = opt.temperature_span
        self.skiplist = {w: True for symbol in string.punctuation for w in [symbol, tokenizer.encode(symbol, add_special_tokens=False)[0]]} 

    def mask(self, tokens):
        mask = [[(x != self.tokenizer.pad_token_id) and (x not in self.skiplist) for x in d] for d in tokens.cpu().tolist()]
        return mask

    def forward(self, q_tokens, q_mask, c_tokens, c_mask, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        CELoss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # multi-vector: (B L H)
        qemb, qembs = self.encoder(input_ids=q_tokens, attention_mask=q_mask, return_multi_vectors=True)
        cemb, cembs = self.encoder(input_ids=c_tokens, attention_mask=c_mask, return_multi_vectors=True)

        if self.norm_doc:
            cemb = torch.nn.functional.normalize(cemb, p=2, dim=-1)
            cembs = torch.nn.functional.normalize(cembs, p=2, dim=-1)
        if self.norm_query:
            qemb = torch.nn.functional.normalize(qemb, p=2, dim=-1)
            qembs = torch.nn.functional.normalize(qembs, p=2, dim=-1)

        ## [Inbatch-s]: dotproduct
        scores_s = torch.einsum("id, jd->ij", qemb / self.tau, cemb)
        loss_0 = CELoss(scores_s, labels)

        predicted_idx = torch.argmax(scores_s, dim=-1)
        accuracy_s = 100 * (predicted_idx == labels).float().mean()
        logs = {'loss_sent': loss_0, 'acc_sent': accuracy_s}

        ## [Inbatch-p]: SumDot
        # multivector fo query only
        # multiplcation: (1 B 1 H) @ (B' 1 H Lq) = (B' B 1 Lq) --> (B'B Lc Lq)
        scores_p = (cemb.unsqueeze(1).unsqueeze(0) @ qembs.permute(0, 2, 1).unsqueeze(1) / self.tau_p).flatten(0, 1)
        qmask = torch.tensor(self.mask(q_tokens), device=q_tokens.device).bool()
        scores_p[~qmask.repeat(bsz, 1).unsqueeze(1)] = 0 # BB 1 Lq    
        scores_p = scores_p.squeeze(1).sum(-1) # BB Lq
        scores_p = scores_p.view(bsz, -1)
        loss_p = CELoss(scores_p, labels)

        predicted_idx = torch.argmax(scores_p, dim=-1)
        accuracy_p = 100 * (predicted_idx == labels).float().mean()
        logs.update({'loss_p': loss_p, 'acc_p': accuracy_p})

        ## [Inbatch-p]: MaxSim
        # multivector fo query/docuemnt
        scores_m = (cembs.unsqueeze(0) @ qembs.permute(0, 2, 1).unsqueeze(1) / self.tau_m).flatten(0, 1)
        cmask = torch.tensor(self.mask(c_tokens), device=c_tokens.device).bool()
        scores_m[~cmask.repeat(bsz, 1)] = -9999 # BB Lc Lq
        scores_m = scores_m.max(1).values.sum(-1)
        scores_m = scores_m.view(bsz, -1) 
        loss_m = CELoss(scores_m, labels)

        predicted_idx = torch.argmax(scores_m, dim=-1)
        accuracy_m = 100 * (predicted_idx == labels).float().mean()
        logs.update({'loss_m': loss_m, 'acc_m': accuracy_m})

        logs.update(self.encoder.additional_log)

        loss = loss_0 * self.opt.alpha + loss_p * self.opt.beta + loss_m * self.opt.gamma 
        return LateInteractionOutput(loss=loss, acc=accuracy_s, logs=logs)

    def get_encoder(self):
        return self.encoder


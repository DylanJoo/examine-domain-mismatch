import torch
import torch.nn as nn
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from transformers.modeling_outputs import BaseModelOutput

@dataclass
class InBatchOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    acc: Optional[Tuple[torch.FloatTensor, ...]] = None
    losses: Optional[Dict[str, torch.FloatTensor]] = None
    q_span: Optional[Tuple[torch.FloatTensor, ...]] = None
    d_span: Optional[Tuple[torch.FloatTensor, ...]] = None

class InBatch(nn.Module):
    def __init__(self, opt, retriever, tokenizer, label_smoothing=False,):
        super(InBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = label_smoothing
        self.tokenizer = tokenizer
        self.encoder = retriever

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix="", **kwargs):

        # this is for random cropping
        if (k_tokens is None) and (k_mask is None):
            return self.forward_bidirectional(q_tokens, q_mask, **kwargs)

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        temperature = 1.0
        scores = torch.einsum("id, jd->ij", qemb / temperature, kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        return {'loss': loss, 'acc': accuracy}

    def forward_bidirectional(self, tokens, mask, **kwargs):
        """this forward function has only one list of texts, each can be either a query or a key
        """ 

        bsz = len(tokens)
        labels = torch.arange(bsz, dtype=torch.long, device=tokens.device).view(-1, 2).flip([1]).flatten().contiguous()

        emb = self.encoder(input_ids=tokens, attention_mask=mask, normalize=self.norm_query)

        temperature = 1.0
        scores = torch.matmul(emb/temperature, emb.transpose(0, 1))
        scores.fill_diagonal_(float('-inf'))

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing) 

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        return {'loss': loss, 'acc': accuracy}


    def get_encoder(self):
        return self.encoder

class InBatchForSplade(InBatch):
    """ Reminder: splade's additional parameters """

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, **kwargs):
        return super().forward(q_tokens, q_mask, k_tokens, k_mask, **kwargs)

class InBatchWithSpan(InBatch):

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        # [objectives] 
        CELoss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        KLLoss = nn.KLDivLoss(reduction='batchmean')

        # [modify]
        qemb, qsemb, qsids = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb, ksemb, ksids = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        # [modify]

        temperature = 1.0
        scores = torch.einsum("id, jd->ij", qemb / temperature, kemb)
        loss_0 = CELoss(scores, labels)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        # [modify] add loss of (q-span, doc)
        sscores_1 = torch.einsum("id, jd->ij", qsemb / temperature, kemb)
        loss_1 = CELoss(sscores_1, labels)

        # [modify] add loss of (query, d-span)
        sscores_2 = torch.einsum("id, jd->ij", qemb / temperature, ksemb)
        loss_2 = CELoss(sscores_2, labels)

        loss = loss_0 + loss_1 + loss_2

        return InBatchOutput(
	        loss=loss, acc=accuracy,
	        losses={'loss_seg': loss_0, 'loss_qspan': loss_1, 'loss_kspan': loss_2},
	        q_span=qsids, d_span=ksids
	)

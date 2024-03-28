import torch
import torch.nn as nn
import torch.nn.functional as F
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
        MSELoss = nn.MSELoss()

        qemb, qsemb, qsids = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb, ksemb, ksids = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        # [sentence]
        temperature = 1.0
        scores = torch.einsum("id, jd->ij", qemb / temperature, kemb)
        loss_0 = CELoss(scores, labels)
        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        # [span]
        if self.opt.distil_from_sentence:
            if self.opt.distil_from_sentence.lower() == 'kl':
                probs_sents = F.softmax(scores, dim=1)
                scores_spans = torch.einsum("id, jd->ij", qsemb / temperature, ksemb)
                logits_spans = F.log_softmax(scores_spans, dim=1)
                loss_span = KLLoss(logits_spans, probs_sents)
                losses = {'loss_sent': loss_0, 'loss_span': loss_span}

            elif self.opt.distil_from_sentence.lower() == 'mse':
                scores_spans = torch.einsum("id, jd->ij", qsemb / temperature, ksemb)
                # loss_span = MSELoss(scores_spans.view(-1), scores.view(-1)) # distil from scores
                loss_span = ( MSELoss(qsemb, qemb) + MSELoss(ksemb, kemb) ) / 2
                losses = {'loss_sent': loss_0, 'loss_span': loss_span}

        else:
            ## add loss of (q-span, doc) ## add loss of (query, d-span)
            sscores_1 = torch.einsum("id, jd->ij", qsemb / temperature, kemb)
            loss_1 = CELoss(sscores_1, labels)
            sscores_2 = torch.einsum("id, jd->ij", qemb / temperature, ksemb)
            loss_2 = CELoss(sscores_2, labels)
            loss_span = (loss_1 + loss_2) / 2
            losses = {'loss_sent': loss_0, 'loss_spans(cont)': loss_span}

        loss = loss_0 + loss_span

        return InBatchOutput(
	        loss=loss, 
	        acc=accuracy,
	        losses=losses,
	        q_span=qsids, 
	        d_span=ksids,
	)

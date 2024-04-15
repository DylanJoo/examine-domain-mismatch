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
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class InBatch(nn.Module):
    def __init__(self, opt, retriever, tokenizer, label_smoothing=False,):
        super(InBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.norm_spans = opt.norm_spans
        self.label_smoothing = label_smoothing
        self.tokenizer = tokenizer
        self.encoder = retriever

        self.tau = opt.temperature
        self.tau_span = opt.temperature_span

    def forward(self, q_tokens, q_mask, c_tokens, c_mask, stats_prefix="", **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask)
        cemb = self.encoder(input_ids=c_tokens, attention_mask=c_mask)
        if self.norm_query:
            qemb = torch.nn.functional.normalize(qemb, p=2, dim=-1)
        if self.norm_doc:
            cemb = torch.nn.functional.normalize(cemb, p=2, dim=-1)

        scores = torch.einsum("id, jd->ij", qemb / self.tau, cemb)
        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        return {'loss': loss, 'acc': accuracy}

    def forward_bidirectional(self, tokens, mask, **kwargs):
        """this forward function has only one list of texts, each can be either a query or a key
        """ 

        bsz = len(tokens)
        labels = torch.arange(bsz, dtype=torch.long, device=tokens.device).view(-1, 2).flip([1]).flatten().contiguous()

        emb = self.encoder(input_ids=tokens, attention_mask=mask)

        scores = torch.matmul(emb/self.tau, emb.transpose(0, 1))
        scores.fill_diagonal_(float('-inf'))

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing) 

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        return {'loss': loss, 'acc': accuracy}

    def get_encoder(self):
        return self.encoder

class InBatchForSplade(InBatch):
    """ Reminder: splade's additional parameters """

    def forward(self, q_tokens, q_mask, c_tokens, c_mask, **kwargs):
        return super().forward(q_tokens, q_mask, c_tokens, c_mask, **kwargs)

class InBatchWithSpan(InBatch):

    def forward(self, q_tokens, q_mask, c_tokens, c_mask, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        # [objectives] 
        CELoss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        KLLoss = nn.KLDivLoss(reduction='batchmean')
        MSELoss = nn.MSELoss()

        # add the dataclass for outputs
        qemb, qsemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask)
        cemb, csemb = self.encoder(input_ids=c_tokens, attention_mask=c_mask)

        if self.norm_query:
            qemb = torch.nn.functional.normalize(qemb, p=2, dim=-1)
            qsemb = torch.nn.functional.normalize(qsemb, p=2, dim=-1)
        if self.norm_doc:
            cemb = torch.nn.functional.normalize(cemb, p=2, dim=-1)
            csemb = torch.nn.functional.normalize(csemb, p=2, dim=-1)

        logs = {}
        # [sentence]
        scores = torch.einsum("id, jd->ij", qemb / self.tau, cemb)
        loss_0 = CELoss(scores, labels)
        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        logs.update({'loss_sent': loss_0, 'acc_sent': accuracy})

        # [span]
        if self.opt.span_sent_interaction == 'kl':
            # distill from scores
            probs_sents = F.softmax(scores, dim=1)
            scores_spans = torch.einsum("id, jd->ij", qsemb / self.tau_span, csemb)
            logits_spans = F.log_softmax(scores_spans, dim=1)
            loss_span = KLLoss(logits_spans, probs_sents)

        elif self.opt.span_sent_interaction == 'cont':
            ## add loss of (q-span, doc) ## add loss of (query, d-span)
            scores_spans = torch.einsum("id, jd->ij", qsemb / self.tau_span, cemb)

            if self.opt.span_span_interaction:
                scores_spans2 = torch.einsum("id, jd->ij", csemb / self.tau_span, qsemb)
                loss_span = CELoss(scores_spans, labels) + CELoss(scores_spans2, labels) 
                logs.update({'acc_span2': accuracy_span2})
            else:
                loss_span = CELoss(scores_spans, labels) 

        predicted_idx = torch.argmax(scores_spans, dim=-1) # check only one as it's prbbly similar
        accuracy_span = 100 * (predicted_idx == labels).float().mean()

        logs.update({'loss_span': loss_span, 'acc_span': accuracy_span})
        logs.update(self.encoder.additional_log)
        loss = loss_0 * self.opt.alpha + loss_span * self.opt.beta

        return InBatchOutput(
	        loss=loss, 
	        acc=accuracy,
          logs=logs,
	)

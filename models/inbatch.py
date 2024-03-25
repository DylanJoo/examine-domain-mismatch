import torch
import torch.nn as nn
import random

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
        # labels = torch.arange(0, bsz, dtype=torch.long, device=tokens.device)
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

        # [modify]
        qemb, qsemb, qsids = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb, ksemb, ksids = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        # [modify]

        temperature = 1.0
        scores = torch.einsum("id, jd->ij", qemb / temperature, kemb)
        loss_0 = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        # [modify] add loss of (q-span, doc)
        sscores_1 = torch.einsum("id, jd->ij", qsemb / temperature, kemb)
        loss_1 = torch.nn.functional.cross_entropy(sscores_1, labels, label_smoothing=self.label_smoothing)

        # [modify] add loss of (query, d-span)
        sscores_2 = torch.einsum("id, jd->ij", qemb / temperature, ksemb)
        loss_2 = torch.nn.functional.cross_entropy(sscores_2, labels, label_smoothing=self.label_smoothing)

        loss = loss_1 + loss_2

        return {'loss': loss, 'acc': accuracy, 
                'loss_1': loss_1, 'loss_2': loss_2,
                'q-span': qsids, 'k-span': ksids}

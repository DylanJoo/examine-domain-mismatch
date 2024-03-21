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

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        # [modify]
        qemb, qsemb, qsids = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb, ksemb, ksids = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        # [modify]

        temperature = 1.0
        scores = torch.einsum("id, jd->ij", qemb / temperature, kemb)
        loss_1 = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        # [modify] add loss
        temperature = 1.0
        sscores = torch.einsum("id, jd->ij", qsemb / temperature, kemb)
        loss_2 = torch.nn.functional.cross_entropy(sscores, labels, label_smoothing=self.label_smoothing)

        loss = loss_1 + loss_2

        return {'loss': loss, 'acc': accuracy, 
                'loss_1': loss_1, 'loss_2': loss_2,
                'q-span': qsids, 'k-span': ksids}

    def get_encoder(self):
        return self.encoder

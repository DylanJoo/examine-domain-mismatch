import torch
import torch.nn as nn
import random
import transformers
import dist_utils

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

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        gather_fn = dist_utils.gather
        gather_kemb = gather_fn(kemb)
        labels = labels + dist_utils.get_rank() * len(kemb)

        temperature = 1.0
        scores = torch.einsum("id, jd->ij", qemb / temperature, gather_kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        # stdq = torch.std(qemb, dim=0).mean().item()
        # stdk = torch.std(kemb, dim=0).mean().item()

        return {'loss': loss, 'acc': accuracy}

    def get_encoder(self):
        return self.encoder

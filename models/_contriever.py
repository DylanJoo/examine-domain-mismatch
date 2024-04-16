import os
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Contriever(BertModel):
    def __init__(self, config, add_pooling_layer=False, pooling='mean', span_pooling=None, **kwargs):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.config.pooling = pooling
        self.config.span_pooling = span_pooling
        self.additional_log = {}

        if self.config.span_pooling:
            if "span_select_weird" in self.config.span_pooling: # may need to add constraints
                self.outputs = nn.Sequential(nn.Linear(self.config.hidden_size, 1), nn.Softmax(1))
            elif "span_extract" in self.config.span_pooling:
                self.outputs = nn.Sequential(nn.Linear(self.config.hidden_size, 2))
        else:
            self.outputs = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_multi_vectors=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True
        )

        last_hidden_states = model_output["last_hidden_state"]
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        # sentence representation
        if self.config.pooling == 'cls':
            emb = last_hidden[:, 0]
        elif self.config.pooling == 'mean':
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        if self.config.span_pooling is None:
            if return_multi_vectors:
                return emb, last_hidden 
            else:
                return emb
        else:
            # sub-sentence representation
            bsz, max_len, hsz = last_hidden.size()
            kwargs = {'hidden': last_hidden, 'mask': attention_mask, 
                      'bsz': bsz, 'max_len': max_len, 'hsz': hsz, 
                      'return_multi_vectors': return_multi_vectors}

            if "span_average" in self.config.span_pooling: 
                span_emb = last_hidden.sum(dim=1) / (attention_mask.sum(dim=1) - 1)[..., None]
            elif "span_key_extract" in self.config.span_pooling: 
                span_emb = self._span_key_extract(**kwargs)
            elif "span_select_weird" in self.config.span_pooling: 
                span_emb = self._span_select_weird(**kwargs)
            elif "span_extract" in self.config.span_pooling: 
                span_emb = self._span_extract(**kwargs)

            return emb, span_emb

    def _span_key_extract(self, hidden, mask, bsz, max_len, return_multi_vectors=False, **kwargs):
        doc_emb = hidden.sum(dim=1) / mask.sum(dim=1)[..., None] # B H
        doc_tok_sim = F.normalize(hidden, p=2, dim=1) @ F.normalize(doc_emb.unsqueeze(-1), p=2, dim=1) # B L H x B 1 H = B L 1
        span_emb = hidden * doc_tok_sim

        if return_multi_vectors:
            return span_emb, doc_tok_sim
        else:
            return span_emb.sum(dim=1) / mask.sum(dim=1)[..., None]

    def _span_extract(self, hidden, mask, bsz, max_len, return_multi_vectors=False, **kwargs):
        logits = self.outputs(hidden)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # hard
        start_probs = nn.functional.gumbel_softmax(start_logits, hard=True)
        end_probs = nn.functional.gumbel_softmax(end_logits, hard=True)
        # start_probs = nn.functional.softmax(start_logits, dim=-1)
        # end_probs = nn.functional.softmax(end_logits, dim=-1)

        start_probs_vec = start_probs.cumsum(-1)
        end_probs_vec = torch.flip(torch.flip(end_probs, [1]).cumsum(-1), [1])
        span_mask = start_probs_vec * end_probs_vec # B L
        span_mask_ = span_mask.clone()
        span_mask_[span_mask==0] = 1e-6

        self.additional_log['extract_ratio'] = ( (span_mask * mask).sum(dim=1) / mask.sum(dim=-1) ).mean()
        self.additional_log['extract_length'] = torch.abs(start_logits.argmax(dim=-1) - end_logits.argmax(dim=-1)).float().mean()

        span_emb = hidden * span_mask_[..., None] # B L H

        if return_multi_vectors:
            return span_emb
        else:
            return span_emb.sum(dim=1) / mask.sum(dim=-1)[..., None]

    def _span_select_weird(self, hidden, mask, return_multi_vectors=False, **kwargs):
        select_prob = self.outputs(hidden) # exclude CLS
        top_k_ids = select_prob.squeeze(-1).topk(10).indices 
        span_emb = hidden * select_prob
        if return_multi_vectors:
            return span_emb
        else:
            return span_emb.sum(dim=1) / mask.sum(dim=1)[..., None]

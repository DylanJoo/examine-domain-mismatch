import os
import torch
import transformers
import torch.nn as nn
from transformers import BertModel


"""
Version 1
 - DPR-reader like start and end token extraction
 - Query: segment representation (from CLS), and span representation
        - (a) `boundary_average`: a start and an end token average as span (SBO)
        - (b) `span_select_average`: a start and an end token concatenation as span (DensePhrase)
        - (c) `span_extract_average`: multiple tokens within the start and end token average as span (retrieval-specific)
        - We will possibly need regularization to make this span as short as possible
 - Document: segment representation (from CLS)

Version 2
 - Binary token classification layer
"""

class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

        if pooling == "cls_boundary_average":
            self.outputs = nn.Linear(self.config.hidden_size, 2)
        elif pooling == "cls_span_select_average":
            self.outputs = nn.sequential(
                    nn.Linear(self.config.hidden_size, 1),
                    nn.Sigmoid()
            )
        elif pooling == "cls_span_extract_average":
            self.outputs = nn.Linear(self.config.hidden_size, 2)
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
        normalize=False,
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
            output_hidden_states=output_hidden_states,
        )

        last_hidden_states = model_output["last_hidden_state"]
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        bsz, seq_len = input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        emb_size = last_hidden.size(-1)

        elif self.config.pooling == "cls_boundary_average":
            emb = last_hidden[:, 0]
            logits = self.outputs(last_hidden_states[:, 1:-1, :]) # exclude CLS and SEP
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
            
            ## resize
            start_id = start_logits.view(bsz, seq_len-2).argmax(-1) # bsz 1
            end_id = end_logits.view(bsz, seq_len-2).argmax(-1)

            failed_span_mask = torch.where(end_id > start_id, 1.0, 0.0)
            span_ids = torch.cat([start_id, end_id], dim=-1).view(2, -1)
            span_emb = last_hidden_states[:, 1:-1, :].gather(
                    1, span_ids.permute(1, 0)[..., None].repeat(1,1,emb_size)
            ).mean(1)
            span_emb = span_emb * failed_span_mask.view(-1, 1)

        elif self.config.pooling == "cls_span_select_average": # may need to add constraints
            emb = last_hidden[:, 0]
            select_prob = self.outputs(last_hidden_states[:, 1:-1, :]) # exclude CLS and SEP
            span_emb = torch.mean(last_hidden_states * select_prob[..., None], dim=1) # bsz L(to 1) H

        elif self.config.pooling == "cls_span_extract_average":
            emb = last_hidden[:, 0]
            logits = self.outputs(last_hidden_states[:, 1:-1, :]) # exclude CLS and SEP
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            ## resize
            start_id = start_logits.view(bsz, seq_len-2).argmax(-1)
            end_id = end_logits.view(bsz, seq_len-2).argmax(-1)
            ordered = torch.arange(seq_len).repeat(bsz).reshape(bsz, seq_len)
            extract_span_mask = (start_id <= ordered) & (ordered <= end_id)

            span_emb = torch.mean(last_hidden_states[:, 1:-1, :] * extract_span_mask.unsqueeze(-1), dim=1)

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)

        return emb, span_emb, span_ids


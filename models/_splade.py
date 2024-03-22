import os
import torch
from transformers import BertForMaskedLM

def normalize(tensor, eps=1e-9):
    """normalize input tensor on last dimension """
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)

def splade_max(tensor, additional_mask=None):
    values, indices = torch.max(
        torch.log(1 + torch.relu(tensor)) * additional_mask.unsqueeze(-1), 
        dim=1
    )
    return values

class SpladeRep(BertForMaskedLM):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling
            self.pooler = splade_max

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

        logits = model_output["logits"] # bsz seq_len V
        vector = logits.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.pooler:
            vector = self.pooler(vector)

        if normalize:
            vector = normalize(vector)
        return vector

import os
import torch
import transformers
from transformers import BertModel



class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

        ## Exp1: span extraction: add start/end token layers for encoder
        ### (a) automatic: distill spans that has higher correlation to the cls embeddings
        self.span_outputs = nn.Linear(self.encoder.embeddings_size, 2)

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

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        elif self.config.pooling == "cls-span_average":
            emb = last_hidden[:, 0]

            # span's embeddings from multivectors
            n_passages, sequence_length = input_ids.size() \
                    if input_ids is not None else inputs_embeds.size()[:2]

            # compute logits 
            ## [note] remove the first CLS token
            ## start/end logits: (B, L, 1).squeeze()
            span_logits = self.span_outputs(last_hidden[:, 1:])
            start_logits, end_logits = span_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).continuous()
            end_logits = end_logits.squeeze(-1).continuous()

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    def forward_span_emb(self, last_hidden_states, start_logits, end_logits):
        start_ids = start_logits.argmax(-1)
        end_ids = end_logits.argmax(-1)
        span_ids = torch.cat([start_ids, end_ids]).view(2, -1)
        span_emb = 0
        return span_emb


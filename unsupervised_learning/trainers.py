from transformers import Trainer
from transformers.utils import logging
from transformers.modeling_utils import unwrap_model
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class TrainerBase(Trainer):

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]


        if self.state.global_step % 10 == 0:
            logger.info(f"loss: {outputs['loss'].item()} | acc: {outputs['acc']}")
            logger.info(f"loss-seg: {outputs['loss_1'].item()} | loss-span: {outputs['loss_2'].item()}")
            logger.info(outputs['q-span'])
            logger.info(outputs['k-span'])

        return (loss, outputs) if return_outputs else loss

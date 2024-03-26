import os
import sys
import json
from typing import Optional, Union
from transformers import HfArgumentParser
from transformers import AutoTokenizer

# from models import Contriever
# from models import InBatch
from trainers import TrainerBase

## [todo] merge these if possible
from ind_cropping.options import ModelOptions, DataOptions, TrainOptions
from ind_cropping.data import load_dataset, Collator

os.environ["WANDB_PROJECT"]="SSL4LEDR"


def main():

    parser = HfArgumentParser((ModelOptions, DataOptions, TrainOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()

    # [Model] tokenizer, model architecture (with bi-encoders)
    tokenizer = AutoTokenizer.from_pretrained(model_opt.model_path or model_opt.model_name)

    # [Model-Dev]
    from models._dev import Contriever 
    encoder = Contriever.from_pretrained(model_opt.model_name, pooling=model_opt.pooling)
    from models.inbatch import InBatchWithSpan
    model = InBatchWithSpan(model_opt, retriever=encoder, tokenizer=tokenizer)
    
    # [Data] train/eval datasets, collator, preprocessor
    train_dataset = load_dataset(data_opt, tokenizer)
    eval_dataset = None
    collator = Collator(opt=data_opt)

    trainer = TrainerBase(
            model=model, 
            args=train_opt,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
    )
    
    # [Training]
    trainer.train(resume_from_checkpoint=train_opt.resume_from_checkpoint)
    trainer.save_model(os.path.join(train_opt.output_dir))

    final_path = train_opt.output_dir
    if  trainer.is_world_process_zero():
        with open(os.path.join(final_path, "model_opt.json"), "w") as write_file:
            json.dump(asdict(model_opt), write_file, indent=4)
        with open(os.path.join(final_path, "data_opt.json"), "w") as write_file:
            json.dump(asdict(data_opt), write_file, indent=4)
        with open(os.path.join(final_path, "train_opt.json"), "w") as write_file:
            json.dump(train_opt.to_dict(), write_file, indent=4)

if __name__ == '__main__':
    main()

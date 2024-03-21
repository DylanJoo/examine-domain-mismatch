import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union
from transformers import TrainingArguments

@dataclass
class ModelOptions:
    model_name: Optional[str] = field(default=None)
    model_path: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    pooling: Optional[str] = field(default="average")
    norm_doc: Optional[bool] = field(default=False)
    norm_query: Optional[bool] = field(default=False)
    output_span: Optional[bool] = field(default=False)

@dataclass
class DataOptions:
    train_data_dir: Optional[str] = field(default='/home/dju/datasets/test_collection/bert-base-uncased')
    eval_data_dir: Optional[str] = field(default=None)
    loading_mode: Optional[str] = field(default="full")
    chunk_length: Optional[int] = field(default=256)
    ratio_min: Optional[float] = field(default=0.1)
    ratio_max: Optional[float] = field(default=0.5)
    augmentation: Optional[str] = field(default=None)

@dataclass
class TrainOptions(TrainingArguments):
    output_dir: str = field(default='./')
    seed: int = field(default=42)
    data_seed: int = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=-1)
    save_steps: int = field(default=1000)
    eval_steps: int = field(default=1000)
    evaluation_strategy: Optional[str] = field(default='no')
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    logging_dir: Optional[str] = field(default='./logs')
    resume_from_checkpoint: Optional[str] = field(default=None)
    save_total_limit: Optional[int] = field(default=2)
    learning_rate: Union[float] = field(default=5e-5)
    remove_unused_columns: bool = field(default=False)
    dataloader_num_workers: int = field(default=1)
    dataloader_prefetch_factor: int = field(default=2)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        """ Add more description for navigating these parameters """
        ## model's parameters
        self.parser.add_argument("--model_name", type=str, default="none", 
            help="the pretrianed huggingface model name")
        self.parser.add_argument("--model_path", type=str, default=None,
            help="the path for retraining/inferencing")
        self.parser.add_argument("--output_dir", type=str, default="./", 
            help="models are saved here")

        ## data's parameters
        self.parser.add_argument("--train_data_dir", type=str, 
                default='/home/dju/datasets/test_collection/bert-base-uncased',
        )
        self.parser.add_argument("--loading_mode", type=str, default="full")
        self.parser.add_argument("--chunk_length", type=int, default=256)
        self.parser.add_argument("--ratio_min", type=float, default=0.1)
        self.parser.add_argument("--ratio_max", type=float, default=0.5)

        ## training parameters of Huggingface Trainer
        self.parser.add_argument("--resume_from_checkpoint", type=str, default=None,
             help="the path to a folder with a valid checkpoint for your model"
        )
        self.parser.add_argument("--optim", type=str, default="adamw")
        self.parser.add_argument("--scheduler", type=str, default="linear")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--lr_min_ratio", type=float, default=0.0,
            help="minimum learning rate at the end of the optimization schedule as a ratio of the learning rate",
        )
        self.parser.add_argument("--weight_decay", type=float, default=0.01, help="learning rate")
        self.parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
        self.parser.add_argument("--beta2", type=float, default=0.98, help="beta2")
        self.parser.add_argument("--eps", type=float, default=1e-6, help="eps")
        self.parser.add_argument("--log_freq", type=int, default=100, 
            help="log train stats every <log_freq> steps during training")
        self.parser.add_argument("--eval_freq", type=int, default=500, 
            help="evaluate model every <eval_freq> steps during training")
        self.parser.add_argument("--save_freq", type=int, default=50000)
        self.parser.add_argument("--maxload", type=int, default=None)
        self.parser.add_argument("--label_smoothing", type=float, default=0.0)
        self.parser.add_argument("--save_only_model", type=float, default=False,
            help="When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."
                 "Note that when this is true, you won't be able to resume training from checkpoint."
                 "This enables you to save storage by not storing the optimizer, scheduler & rng state."
                 "You can only load the model using from_pretrained with this option set to True.")
        self.parser.add_argument("--skip_memory_metrics", default=True, 
            help="Whether or not to skip adding of memory profiler reports to metrics.")

        ## additional parameters from huggingface trainer's side
        self.parser.add_argument("--full_determinism", type=bool, default=False,
            help="Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
            "training. Important: this will negatively impact the performance, so only use it for debugging.")
        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")
        ### Do not touch this type annotation or it will stop working in CLI
        self.parser.add_argument("--accelerator_config", type=str, default=None,
            help="Config to be used with the internal Accelerator object initializtion. The value is either a "
             "accelerator json config file (e.g., `accelerator_config.json`) or an already loaded json file as `dict`.")
        self.parser.add_argument("--deepspeed", type=str, default=None,
            help="Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
             " loaded json file as a dict")
        ### actually this is not appeared in huggingface
        self.parser.add_argument("--deepspeed_plugin", type=str, default=None)

        # basic parameters
        self.parser.add_argument(
            "--eval_data",
            nargs="+",
            default=[],
            help="Data used for evaluation during finetuning, this option is not used during contrastive pre-training.",
        )
        self.parser.add_argument(
            "--eval_datasets", nargs="+", default=[], help="List of datasets used for evaluation, in BEIR format"
        )
        self.parser.add_argument(
            "--eval_datasets_dir", type=str, default="./", help="Directory where eval datasets are stored"
        )
        self.parser.add_argument("--continue_training", action="store_true")
        self.parser.add_argument("--num_workers", type=int, default=5)

        self.parser.add_argument("--lower_case", action="store_true", help="perform evaluation after lowercasing")
        self.parser.add_argument(
            "--sampling_coefficient",
            type=float,
            default=0.0,
            help="coefficient used for sampling between different datasets during training, \
                by default sampling is uniform over datasets",
        )
        self.parser.add_argument("--augmentation", type=str, default="none")
        self.parser.add_argument("--prob_augmentation", type=float, default=0.0)

        self.parser.add_argument("--dropout", type=float, default=0.1)
        self.parser.add_argument("--rho", type=float, default=0.05)

        self.parser.add_argument("--contrastive_mode", type=str, default="moco")
        self.parser.add_argument("--queue_size", type=int, default=65536)
        self.parser.add_argument("--temperature", type=float, default=1.0)
        self.parser.add_argument("--momentum", type=float, default=0.999)
        self.parser.add_argument("--moco_train_mode_encoder_k", action="store_true")
        self.parser.add_argument("--eval_normalize_text", action="store_true")
        self.parser.add_argument("--norm_query", action="store_true")
        self.parser.add_argument("--norm_doc", action="store_true")
        self.parser.add_argument("--projection_size", type=int, default=768)

        self.parser.add_argument("--score_function", type=str, default="dot")
        self.parser.add_argument("--retriever_model_id", type=str, default="bert-base-uncased")
        self.parser.add_argument("--pooling", type=str, default="average")
        self.parser.add_argument("--random_init", action="store_true", help="init model with random weights")

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=2, type=int, help="Batch size per GPU for training.")
        self.parser.add_argument(
            "--per_gpu_eval_batch_size", default=2, type=int, help="Batch size per GPU for evaluation."
        )
        self.parser.add_argument("--total_steps", type=int, default=1000)
        self.parser.add_argument("--warmup_steps", type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=10001, help="Master port (for multi-node SLURM jobs)")
        self.parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")

        # finetuning options
        self.parser.add_argument("--negative_ctxs", type=int, default=1)
        self.parser.add_argument("--negative_hard_min_idx", type=int, default=0)
        self.parser.add_argument("--negative_hard_ratio", type=float, default=0.0)

    def print_options(self, opt):
        message = ""
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: %s]" % str(default)
            message += f"{str(k):>40}: {str(v):<40}{comment}\n"
        print(message, flush=True)
        model_dir = os.path.join(opt.output_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(os.path.join(opt.output_dir, "models"))
        file_name = os.path.join(opt.output_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        opt, _ = self.parser.parse_known_args()
        # opt = self.parser.parse_args()
        return opt

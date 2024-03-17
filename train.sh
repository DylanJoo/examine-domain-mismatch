torchrun --nproc_per_node 2 \
    unsupervised_learning/train.py \
    --model_name facebook/contriever \
    --output_dir models/ckpt/test \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --train_data_dir /home/dju/datasets/test_collection/bert-base-uncased

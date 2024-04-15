#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=retrieval
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.%j.debug

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.
# cd examine-domain-mismatch

bash run_beir_dr.sh

#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=create
#SBATCH --partition cpu
#SBATCH --mem=15G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

for dataset in trec-covid;do
    NSPLIT=2 #Must be larger than the number of processes used during training
    OUTDIR=${HOME}/datasets/beir/${dataset}/ind_cropping
    NPROCESS=1
    INFILE=${HOME}/datasets/beir/${dataset}/collection/corpus.jsonl
    mkdir -p ${OUTDIR}

    split -a 2 -d -n l/${NSPLIT} ${INFILE} ${INFILE}

    pids=()

    for ((i=0;i<$NSPLIT;i++));do
        num=$(printf "%02d\n" $i);
        FILE=${INFILE}${num};
        echo $FILE

        python unsupervised_learning/ind_cropping/preprocess.py \
            --tokenizer bert-base-uncased \
            --datapath ${FILE} \
            --overwrite \
            --outdir ${OUTDIR} &

        pids+=($!);
        if (( $i % $NPROCESS == 0 ))
        then
            for pid in ${pids[@]}; do
                wait $pid
            done
        fi
    done

    for pid in ${pids[@]}; do
        wait $pid
    done

    rm -r ${INFILE}??
    echo done
done

#!/bin/bash
#$ -N lm-gen
#$ -q gpu.q
#$ -l m_mem_free=30G
#$ -j y  
#$ -o job_output/$JOB_NAME-$JOB_ID-$TASK_ID.log
#$ -t 1-400

module load python/3.11.5
source $HOME/venv3115/bin/activate

model='llama-2-13b-chat-hf'
dataset='triviaqa'
bsize=20

python3 -u pipeline/generate --model $model --dataset $dataset --batch_size $bsize --idx $SGE_TASK_ID
python3 -u dataeval/load_run.py --batch_size $bsize --data $data --model $model --idx $SGE_TASK_ID
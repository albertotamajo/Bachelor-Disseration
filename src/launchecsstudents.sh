#!/bin/bash -l
#SBATCH --partition=ecsstudents
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=at2n19@soton.ac.uk



module load conda/py3-latest
conda activate cc3dvaewgan2





python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=8889 trainCC3DVAEWGAN.py
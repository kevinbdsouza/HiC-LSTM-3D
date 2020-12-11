#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --account=rrg-maxwl
#SBATCH --mem=4000M
#SBATCH --mail-user=kevin@ece.ubc.ca
#SBATCH --mail-type=ALL

python train_hic.py


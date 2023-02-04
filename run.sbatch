#!/bin/bash
#SBATCH --job-name=normal
#SBATCH --mem=200M
#SBATCH --time=50
#SBATCH --output=%x-%j.out

module purge
module load GCCcore/11.3.0
module load Python/3.10.4

python ../lr_se0.py > data.off
python ../lr_se1.py > data.on

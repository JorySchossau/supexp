#!/bin/bash
#SBATCH --job-name=normal
#SBATCH --mem=200M
#SBATCH --time=20
#SBATCH --output=%x-%j.out

module purge
module load GCC

mkdir -p ${JOBID}
cd ${JOBID}
cp ../poet .
./poet
echo ${JOBID} > done.txt

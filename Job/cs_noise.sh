#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=16
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH -J ReconFG
#SBATCH -o out/reconst2.out
#SBATCH -e out/reconst2.err
#SBATCH --time=00:15:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate cmblens
cd /global/u2/l/lonappan/workspace/LBlens


mpirun -np $SLURM_NTASKS python cs_noise.py

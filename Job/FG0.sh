#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=8
#SBATCH --ntasks=250
#SBATCH --cpus-per-task=1
#SBATCH -J LBIRD-FG0
#SBATCH -o out/fg0.out
#SBATCH -e out/fg0.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it



#source /global/homes/l/lonappan/.bashrc
module load python
conda activate cmblens
cd /global/u2/l/lonappan/workspace/dell

export ini=LB_FG0.ini


mpirun -np $SLURM_NTASKS python simulation.py $ini -maps
#mpirun -np $SLURM_NTASKS python simulation.py $ini -fg
#mpirun -np $SLURM_NTASKS python simulation.py $ini -noise
#mpirun -np $SLURM_NTASKS python filtering.py $ini -cinv
#mpirun -np $SLURM_NTASKS python quest.py $ini -qlms
#mpirun -np $SLURM_NTASKS python quest.py $ini -N0
#mpirun -np $SLURM_NTASKS python quest.py $ini -qlms_input
#mpirun -np $SLURM_NTASKS python quest.py $ini -resp
#mpirun -np $SLURM_NTASKS python quest.py $ini -RDN0
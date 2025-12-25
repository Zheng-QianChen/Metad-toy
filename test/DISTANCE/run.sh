#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=00:05:00


source ~/zqc/.bashrc
#module purge
#module load cmake/3.31.8
#module load plumed/2.9.3-for-dpmd-excelsior
#export lmp='/home/excelsior/apps/_compile/lammps-29Aug2024/src/lmp_mpi'

RANDOM_SEED=$(date +%s%N | cut -b 11-18) 


mpirun -np 1 lmp -var seed ${RANDOM_SEED} -in in.test-2atoms-self-potential
python plot_hills.py in.test-2atoms-toys HILLS

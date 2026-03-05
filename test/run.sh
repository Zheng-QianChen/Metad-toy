#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=00:05:00


# source ~/zqc/.bashrc
#module purge
#module load cmake/3.31.8
#module load plumed/2.9.3-for-dpmd-excelsior

RANDOM_SEED=$(date +%s%N | cut -b 11-18) 
METAD_PLUGIN_PATH=~/work/Metad-toy/build


#mpirun -np 8 lmp -var seed ${RANDOM_SEED} -in in.test-steinhardt
mpirun -np 1 lmp \
    -var seed ${RANDOM_SEED} \
    -var METAD_PLUGIN_PATH ${METAD_PLUGIN_PATH} \
    -in in.test-steinhardt
python plot_hills.py in.test-steinhardt HILLS
#mpirun -np 1 lmp -var seed ${RANDOM_SEED} -in in.test-2atoms-toys
#python plot_hills.py in.test-2atoms-toys HILLS

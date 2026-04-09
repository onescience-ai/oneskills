#!/bin/bash
#SBATCH -p {cluster.partition}
#SBATCH -N {cluster.nodes}
#SBATCH --gres={cluster.gpu_type}:{cluster.gpus_per_node}
#SBATCH --cpus-per-task={cluster.cpus_per_task}
#SBATCH --ntasks-per-node={cluster.ntasks_per_node}
#SBATCH -J {job_name}
#SBATCH --time={cluster.time_limit}
#SBATCH -o logs/%j.out
#SBATCH --exclusive

echo "START TIME: $(date)"
module purge

##### if DCU: Launch DCU ENV #####
{modules_config}

##### python always Launch Conda ENV #####

source ~/.bashrc
conda activate {conda.env_name}

##### onescience datasets and models Launch env #####
export ONESCIENCE_DATASETS_DIR="{env_vars.ONESCIENCE_DATASETS_DIR}"
export ONESCIENCE_MODELS_DIR="{env_vars.ONESCIENCE_MODELS_DIR}"

##### Show env #####
which python

##### Set DCU #####
export HIP_VISIBLE_DEVICES={hip_visible_devices}

export OMP_NUM_THREADS={cluster.cpus_per_task}
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# 第一个节点的地址
export MASTER_ADDR=$(hostname)

# 在每个节点上启动 torchrun
echo SLURM_NNODES=$SLURM_NNODES
echo "Nodes: ${nodes_array[*]}"
echo SLURM_NTASKS=$SLURM_NTASKS


python {script.code_path}

#!/bin/bash
#SBATCH -e err-diffreg-%J.err
#SBATCH -o out-diffreg-%J.out
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --mail-user=shiliyuandd@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# 切换到你的项目目录
cd /home/shiliyuan/Projects/DiffReg/diffreg_pointnet_trans/

# 直接调用指定环境下的 python
/mnt/cluster/environments/shiliyuan/miniconda3/envs/devtorch118/bin/python train_ddpm2_posi.py


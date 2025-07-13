#!/bin/bash

#export NCCL_DEBUG=info
#export NCCL_P2P_DISABLE=1
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL

source /home/apinyol/anaconda3/etc/profile.d/conda.sh
conda activate latent_intrinsics

while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gn  oded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
data_path=/home/apinyol/TFM/Data/RSR_256

python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=${port} main_relight.py \
--data_path ${data_path} \
--load_ckpt /home/apinyol/TFM/Latent_Intrinsics/TFM/checkpoints/normal_loss_bo_20250709_182906_cf12c53a/latest.pth \


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
data_path=/home/apinyol/TFM/Data/RSR_256 #_mini
#/home/apinyol/TFM/Data/RSR_256
#/home/apinyol/TFM/Data/iiw-dataset/data
#/home/apinyol/TFM/Data/multi_illumination_train_mip2_jpg

python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=${port} main_cls.py \
--data_path ${data_path} \
--reg_weight 1e-4 \
--intrinsics_loss_weight 1e-1 \
--epochs 50 \
--batch_size 16 \
--learning_rate 2e-4 \
--weight_decay 1e-2 \
--wandb_project "Latent_Intrinsics_Relighting" \
--wandb_run_name "rsr finetuning + depth loss" \
--dataset rsr_256 \
--experiment_name "rsr_f_dl" \
--enable_depth_loss \
--depth_loss_weight 10.0 \
--resume_from /home/apinyol/TFM/Models/Latent_Intrinsics/last.pth.tar \

#--resume_from /home/apinyol/TFM/Models/Latent_Intrinsics/last.pth.tar \

# Start new experiment
#--experiment_name "depth_ablation" --learning_rate 0.001
#--experiment_name "rsr_f_dl" dataset_scratch(s)/finetuning(f)_depthloss(dl)/no_depthloss(ndl) 

# Resume from latest checkpoint automatically  
#--experiment_name "depth_ablation" --auto_resume

# Resume from specific checkpoint
#--resume_from ./checkpoints/intrinsics_experiment_20250618_143052_a1b2c3d4/epoch_0050.pth

# Keep only last 5 checkpoints, save every 10 epochs
#--keep_last 5 --save_every 10


# Depth params
# --enable_depth_loss \
# --depth_loss_weight 1e-2 \
# --depth_model_name "depth-anything/Depth-Anything-V2-Small-hf
 
 #wandb projecs: 
# "intrinsic relighting2" \
#"Latent_Intrinsics_Relighting" \
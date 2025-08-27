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
data_path=/data/storage/datasets/RSR_256
#/data/storage/datasets/RSR_256
#/data/storage/datasets/RLSID
#/home/apinyol/TFM/Data/RSR_256_mini
#/home/apinyol/TFM/Data/iiw-dataset/data
#/home/apinyol/TFM/Data/multi_illumination_train_mip2_jpg

python inference.py \
--checkpoint_path /home/apinyol/TFM/TFM/Latent_intrinsics/checkpoints/normals_decoder_loss_arreglat_20250818_021042_b5d56be4/epoch_0050.pth.tar \
--dataset rsr_256 \
--data_path ${data_path} \
--save_images \
--output_dir ./inference_results/NormalsDecoder_frsr256_test_v2 \
--batch_size 256 \
--save_every 50 \
#--max_samples 300
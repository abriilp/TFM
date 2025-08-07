import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument("--num_gpus",type=int,default=1,help = "Number of GPUs to use for training")

# Training parameters
parser.add_argument('--epochs', type=int, default=300, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=2,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')
parser.add_argument('--patch_size', type=int, default=384, help='patchsize of input.')
parser.add_argument('--resize_width', type=int, default=600, help='resize width of input.')
parser.add_argument("--wblogger",type=str,default="promptnorm",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")

# Training path
parser.add_argument('--train_input_dir', type=str, default='/data/storage/datasets/RLSID_promptnorm_100k/train/input', help='training input data path.')
parser.add_argument('--train_normals_dir', type=str, default='/data/storage/datasets/RLSID_promptnorm_100k/train/normals', help='training depth data path.')
parser.add_argument('--train_target_dir', type=str, default='/data/storage/datasets/RLSID_promptnorm_100k/train/target', help='training target data path.')

# Testing / Inference path
parser.add_argument('--test_input_dir', type=str, default='/data/storage/datasets/RLSID_promptnorm_100k/test/input', help='test input data path.')
parser.add_argument('--test_normals_dir', type=str, default='/data/storage/datasets/RLSID_promptnorm_100k/test/normals', help='test depth data path.')
parser.add_argument('--test_target_dir', type=str, default='/data/storage/datasets/RLSID_promptnorm_100k/test/target', help='test target data path.')
parser.add_argument('--test_ref_dir', type=str, default='/data/storage/datasets/RLSID_promptnorm_100k/test/target', help='test target data path.')
parser.add_argument('--pretrained_ckpt_path', type=str, default='/home/apinyol/TFM/Models/promptnorm.ckpt', help='pretrained checkpoint path.')

# Inference and testing output path
parser.add_argument('--output_path', type=str, default="output/", help='output save path')

# Wandb params
parser.add_argument('--wandb_run_name', type=str, default="PromptNorm-Train-Viz", help='Wandb run name')

options = parser.parse_args()


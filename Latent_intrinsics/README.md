# Official Code for [Latent Intrinsics Emerge from Training to Relight (NeurIPS 2024, Spotlight)](https://arxiv.org/abs/2405.21074)

## Training

To train the model, download the [Multi-illumination dataset](https://projects.csail.mit.edu/illumination/) and update the `data_path` in `srun.sh`. Then, use `bash srun.sh` to launch the training script for 4 GPUs.

## Evaluation

### 1. Pretrained Checkpoint

We provide the pre-trained model to infer the albedo.  
Download the pretrained relighting model from this [Google Drive Link](https://drive.google.com/file/d/1bb4Up7SNZ9lBTku4LGAJe49wE4bEVlBo/view?usp=sharing).

### 2. Albedo Evaluation

- Download the [IIW dataset](http://opensurfaces.cs.cornell.edu/publications/intrinsic/).
- Update the `data_path` and `checkpoint_path` in `srun_albedo.sh`.
- Infer the albedo and calculate WHDR by sweeping the threshold with `bash srun_albedo.sh`.

### 3. Relight Evaluation

- Update the `data_path` and `checkpoint_path` in `srun_relight.sh`.
- Run `bash srun_albedo.sh` to evaluate the relighting images with arbitrary references.

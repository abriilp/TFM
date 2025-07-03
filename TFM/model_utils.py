import torch, pdb
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import numpy as np
from PIL import Image
F = torch.nn.functional

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def compute_logdet(W, eps = 0.2):
    m, p = W.shape  # [d, B] 
    W = W.float()
    I = torch.eye(p, device=W.device)
    scalar = p / (m * eps)
    cov = W.T.matmul(W)
    cov = torch.stack(FullGatherLayer.apply(cov))
    cov = cov.mean(dim = 0)
    logdet = torch.logdet(I + scalar * cov)
    return logdet / 2.

def intrinsic_loss(intrinsic1_list, intrinsic2_list):
    sim_intrinsic_list = []
    for intrinsic1, intrinsic2 in zip(intrinsic1_list, intrinsic2_list):
        sim_intrinsic_list.append(((intrinsic1 - intrinsic2)**2).sum(dim = 1).mean())
        #print(f"intrinsic1 shape: {intrinsic1.sum(dim=1).shape}, intrinsic2 shape: {intrinsic2.shape})")
    return torch.stack(sim_intrinsic_list).mean()

class compute_logdet_loss():
    def __init__(self):
        self.target_val_list = None

    def __call__(self, data_list, prints=False):
        logdet_list = []
        target_val_list  = []
        with torch.cuda.amp.autocast(enabled = False):
            for idx, data in enumerate(data_list):
                if len(data.shape) == 4:
                    #print(f"data shape: {data.shape}")
                    data = data.permute(0,2,3,1).reshape(-1, data.shape[1])
                logdet = compute_logdet(data)
                logdet_list.append(logdet)
                if prints:
                        print(f"Logdet {idx}: {logdet_list[idx].item()}")
                if self.target_val_list is None:
                    logdet_target = compute_logdet(F.normalize(torch.randn_like(data), dim = -1)).detach().data
                    target_val_list.append(logdet_target)
                if self.target_val_list is None:
                    if prints:
                        print(f"Logdet target {idx}:  {target_val_list}")
                    self.target_val_list = target_val_list
            return torch.stack(logdet_list), torch.stack(self.target_val_list)

@torch.no_grad()
def plot_relight_img_train(model, input_img, ref_img, target_img, save_path, eval_pair_shift = 2):
    model.eval()

    img1 = input_img
    img2 = ref_img
    img3 = target_img

    intrinsic1, extrinsic1 = model(img1, run_encoder = True)
    intrinsic2, extrinsic2 = model(img2, run_encoder = True)
    intrinsic3, extrinsic3 = model(img3, run_encoder = True)

    with torch.no_grad():
        # Reconstruction
        edm_gen_img_e1_i1 = model([intrinsic1, extrinsic1], run_encoder = False)[:25]

    with torch.no_grad():
        # Relighting with extrinsics inferred from the reference
        edm_gen_img_e2_i1 = model([intrinsic1, extrinsic2], run_encoder = False)[:25]

    with torch.no_grad():
        # Relighting with target extrinsic
        edm_gen_img_e3_i1 = model([intrinsic1, extrinsic3], run_encoder = False)[:25]

    def save_img(img_list, name):
        white_space = (np.ones((eval_pair_shift*256, 20, 3)).astype(np.float32) * 255).astype(np.uint8)
        np_img_list = []
        for img in img_list:
            img = ((img[:25].clamp(-1,1) * 0.5 + 0.5).reshape(eval_pair_shift, eval_pair_shift, 3, 256, 256).permute(0,3, 1, 4, 2).reshape(eval_pair_shift*256, eval_pair_shift*256, 3) * 255).cpu().data.numpy().astype(np.uint8)
            np_img_list.append(img)
            np_img_list.append(white_space)
        Image.fromarray(np.concatenate(np_img_list, axis = 1)).save(f'{name}.png')
    save_img([img1, img2, img3, edm_gen_img_e1_i1, edm_gen_img_e2_i1, edm_gen_img_e3_i1], save_path+'_tot')
    save_img([img1, img3, edm_gen_img_e3_i1], save_path+'_imgs13')
    save_img([img1, img2, edm_gen_img_e2_i1], save_path+'_imgs12')

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

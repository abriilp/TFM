# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch,pdb
from torch.nn.functional import silu
import torch.nn.functional as F
import time
#from resnet import resnet34
import torch.nn as nn
#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

#encoder.cuda()
#encoder(torch.randn(10, 3, 128, 128).float().cuda())

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None, affine_scale = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.affine_scale = affine_scale
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        if emb_channels > 0:
            self.affine = Linear(in_features=emb_channels, out_features=out_channels, **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb = None):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        if emb is not None:
            params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype).tanh() * self.affine_scale
            x = silu(self.norm1(x) * (params + 1))
        else:
            x = silu(self.norm1(x))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

class UNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks_list          = [3,3,3,3],            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        sigma_data = 0.5,
        affine_scale = 1,
        normal_channels = 3                 # Number of channels for normal map output (typically 3 for RGB normals)
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        assert len(num_blocks_list) == len(channel_mult)
        self.sigma_data = sigma_data
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=0, num_heads=1, dropout=dropout, skip_scale=1, eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn, affine_scale = affine_scale
        )
        
        # Store architecture parameters
        self.img_resolution = img_resolution
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_blocks_list = num_blocks_list
        self.attn_resolutions = attn_resolutions
        self.normal_channels = normal_channels
        
        # Encoder (shared)
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks_list[level]):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                if idx == num_blocks_list[level] - 1 and res in [128, 64, 32, 16, 8]:
                    self.enc[f'{res}x{res}_block{idx}_final'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
                else:
                    self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        
        # Positional embedding
        res = img_resolution >> (len(channel_mult) - 1)
        img_edge_embed = PositionalEmbedding(cout // 2)(torch.arange(res) - (res - 1)/ 2)
        pos_embed = torch.cat([img_edge_embed[:,None].expand(-1,res,-1), img_edge_embed[None,:].expand(res, -1, -1)], dim = -1).permute(2,0,1)[None,...]
        self.register_buffer('pos_embed', pos_embed)

        # Light encoding/decoding
        self.light_encoder = nn.Sequential(nn.Linear(cout, cout), nn.SiLU(), nn.Linear(cout, 16))
        self.light_decoder = nn.Sequential(nn.Linear(16, emb_channels), nn.SiLU(), nn.Linear(emb_channels, emb_channels))
        
        # Main decoder (for relighting) - uses light embedding
        self.dec = torch.nn.ModuleDict()
        main_block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=1, eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn, affine_scale = affine_scale
        )
        
        # Build main decoder
        cout_start = cout
        skips = [block.out_channels for name, block in self.enc.items() if 'final' in name]
        for level, (mult, num_blocks) in reversed(list(enumerate(zip(channel_mult, num_blocks_list)))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **main_block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **main_block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **main_block_kwargs)
            if len(skips) > 0:
                skip = skips.pop()
            else:
                skip = 0
            for idx in range(num_blocks + 1):
                cin = cout + skip
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **main_block_kwargs)
        
        self.final = nn.Sequential(
            GroupNorm(num_channels=cout, eps=1e-6), 
            nn.SiLU(), 
            Conv2d(in_channels=cout, out_channels=out_channels, kernel=3)
        )

        # Normal decoder (for normals) - no light embedding
        self.normal_dec = torch.nn.ModuleDict()
        normal_block_kwargs = dict(
            emb_channels=0, num_heads=1, dropout=dropout, skip_scale=1, eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn, affine_scale = affine_scale
        )
        
        # Build normal decoder
        cout = cout_start
        skips_normal = [block.out_channels for name, block in self.enc.items() if 'final' in name]
        for level, (mult, num_blocks) in reversed(list(enumerate(zip(channel_mult, num_blocks_list)))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.normal_dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **normal_block_kwargs)
                self.normal_dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **normal_block_kwargs)
            else:
                self.normal_dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **normal_block_kwargs)
            if len(skips_normal) > 0:
                skip = skips_normal.pop()
            else:
                skip = 0
            for idx in range(num_blocks + 1):
                cin = cout + skip
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.normal_dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **normal_block_kwargs)
        
        self.normal_final = nn.Sequential(
            GroupNorm(num_channels=cout, eps=1e-6), 
            nn.SiLU(), 
            Conv2d(in_channels=cout, out_channels=normal_channels, kernel=3)
        )

    def update_affine_scale(self, affine_scale):
        for layer in self.enc.values():
            if isinstance(layer, UNetBlock):
                layer.affine_scale = affine_scale

        for layer in self.dec.values():
            if isinstance(layer, UNetBlock):
                layer.affine_scale = affine_scale
        
        for layer in self.normal_dec.values():
            if isinstance(layer, UNetBlock):
                layer.affine_scale = affine_scale

    def forward(self, input, run_encoder=False, decode_mode='both'):
        """
        Forward pass with unified interface.
        
        Args:
            x: Input tensor
            decode_mode: 'both', 'main', 'normal', or 'encode_only'
        
        Returns:
            If decode_mode == 'both': (main_output, normal_output)
            If decode_mode == 'main': main_output
            If decode_mode == 'normal': normal_output  
            If decode_mode == 'encode_only': (skips, light_code)
        """
        if run_encoder:
            x = input
            skips = []
            for name, block in self.enc.items():
                if 'final' in name:
                    skips.append(x)
                x = block(x)
                if 'final' in name:
                    skips[-1] = F.normalize(x, dim=1)
            
            # Light encoding
            light_code = F.normalize(self.light_encoder(x.mean(dim=[2,3])), dim=-1)
            
        
            return skips, light_code
        
        # Prepare for decoding
        outputs = {}
        
        if decode_mode in ['both', 'main']:
            input2 = input.copy()
            skips, light_code = input2 # Unpack skips and light_code
            # Main decoder
            light_emb = self.light_decoder(light_code * np.sqrt(light_code.shape[-1]))
            x_main = self.pos_embed.expand(light_code.shape[0], -1, -1, -1)
            skips_main = skips.copy()
            
            for name, block in self.dec.items():
                if ('in1' in name or 'up' in name) and len(skips_main) > 0:
                    skip = skips_main.pop()
                    skip = skip * np.sqrt(skip.shape[1])
                if x_main.shape[1] != block.in_channels:
                    x_main = torch.cat([x_main, skip], dim=1)
                x_main = block(x_main, light_emb)
            
            outputs['main'] = self.final(x_main)
        
        if decode_mode in ['both', 'normal']:
            skips, light_code = input
            # Normal decoder
            x_normal = self.pos_embed.expand(skips[-1].shape[0], -1, -1, -1)
            skips_normal = skips.copy()
            
            for name, block in self.normal_dec.items():
                if ('in1' in name or 'up' in name) and len(skips_normal) > 0:
                    skip = skips_normal.pop()
                    skip = skip * np.sqrt(skip.shape[1])
                if x_normal.shape[1] != block.in_channels:
                    x_normal = torch.cat([x_normal, skip], dim=1)
                x_normal = block(x_normal)
            
            outputs['normal'] = self.normal_final(x_normal)
        
        # Return based on decode_mode
        if decode_mode == 'both':
            return outputs['main'], outputs['normal']
        elif decode_mode == 'main':
            return outputs['main']
        elif decode_mode == 'normal':
            return outputs['normal']
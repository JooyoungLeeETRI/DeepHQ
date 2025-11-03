from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

import torchac

from einops import rearrange 
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math

from .utils import get_boundary_list, get_cdf_list, get_reconstructions

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

#for debug
def register_backward_hooks(module, name_prefix=""):
    def hook_fn(module, grad_input, grad_output):
        for i, gi in enumerate(grad_input):
            if gi is not None and (torch.isnan(gi).any() or torch.isinf(gi).any()):
                print(f"[{name_prefix}] ⚠️ NaN/Inf in grad_input[{i}]")
        for i, go in enumerate(grad_output):
            if go is not None and (torch.isnan(go).any() or torch.isinf(go).any()):
                print(f"[{name_prefix}] ⚠️ NaN/Inf in grad_output[{i}]")
    module.register_full_backward_hook(hook_fn)


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x

class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out

# class SWAtten(AttentionBlock):
#     def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
#         if inter_dim is not None:
#             super().__init__(N=inter_dim)
#             self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
#         else:
#             super().__init__(N=input_dim)
#             self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)

#         if inter_dim is not None:
#             self.in_conv = conv1x1(input_dim, inter_dim)
#             self.out_conv = conv1x1(inter_dim, output_dim)

#     def _register_nan_inf_hook(self, name, tensor):
#         if tensor.requires_grad:
#             def hook_fn(grad):
#                 if torch.isnan(grad).any() or torch.isinf(grad).any():
#                     print(f"[Gradient Anomaly Detected] in {name}")
#                     print(f"  min: {grad.min().item()}, max: {grad.max().item()}, mean: {grad.mean().item()}")
#                     print(f"  grad shape: {grad.shape}")
#             tensor.register_hook(hook_fn)

#     def forward(self, x):
#         x = self.in_conv(x)
#         self._register_nan_inf_hook("in_conv_out", x)

#         identity = x
#         z = self.non_local_block(x)
#         self._register_nan_inf_hook("non_local_block_out", z)

#         a = self.conv_a(x)
#         b = self.conv_b(z)

#         self._register_nan_inf_hook("conv_a_out", a)
#         self._register_nan_inf_hook("conv_b_out", b)

#         out = a * torch.sigmoid(b)
#         out += identity
#         out = self.out_conv(out)
#         self._register_nan_inf_hook("out_conv_out", out)
#         return out

class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='W')
        self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='SW')
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x =  self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x

# class ImportanceMap(nn.Module):
#     def __init__(self, M):
#         super().__init__()
#         self.proj = nn.Conv2d(2 * M, M, kernel_size=1)

#     def forward(self, x):
#         x = self.proj(x)
#         # x = x + 0.5
#         # x = torch.clamp(x, min=0.0, max=1.0)
#         x = torch.sigmoid(torch.clamp(x, -20.0, 20.0))
#         # x = torch.sigmoid(x)
#         return x

class ImportanceMap(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(N, N, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(N, N * 3 // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(N * 3 // 2,M, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.conv_out(x)
        x += 5.0
        x = torch.sigmoid(torch.clamp(x, -20.0, 20.0))
        return x
    
        

class DebugSigmoid(nn.Module):
    def forward(self, x):
        if torch.isnan(x).any():
            print("[NaN Detected] Input to Sigmoid")
        out = torch.sigmoid(torch.clamp(x, -20.0, 20.0))
        if torch.isnan(out).any():
            print("[NaN Detected] Output from Sigmoid")
        return out

class TCM_Phase3(CompressionModel):
    def __init__(self, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=128,  M=320, num_slices=1, max_support_slices=5, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[2])] + \
                      [conv3x3(2*N, M, stride=2)]

        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[4])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[5])] + \
                      [subpel_conv3x3(2*N, 3, 2)]
        
        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, 2*N, 2)] + self.m_down1 + self.m_down2 + self.m_down3)
        

        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, 2*N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.ha_down1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [conv3x3(2*N, 192, stride=2)]

        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride(320, 2*N, 2)] + self.ha_down1
        )

        self.hs_up1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320, 2)]

        self.h_mean_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2*N, 2)] + self.hs_up1
        )

        self.hs_up2 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320, 2)]


        self.h_scale_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2*N, 2)] + self.hs_up2
        )

        self.hs_up3 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320, 2)]
        
        # self.h_importance_s = nn.Sequential(
        #     *([ResidualBlockUpsample(192, 2*N, 2)] + list(self.hs_up3) + [nn.Sigmoid()])
        # )
        # self.h_importance_s = nn.Sequential(
        #     *[ResidualBlockUpsample(192, 2*N, 2)] + self.hs_up3 + [DebugSigmoid()]
        # )
        
        

        # self.importance_map = ImportanceMap(M)
        self.importance_map = ImportanceMap(192,M)


        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )
        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )
        
        #for debug
        # for i, seq in enumerate(self.cc_scale_transforms):
        #     for j, layer in enumerate(seq):
        #         if isinstance(layer, (nn.Conv2d, nn.GELU)):
        #             register_backward_hooks(layer, name_prefix=f"cc_scale_transforms[{i}][{layer.__class__.__name__}_{j}]")

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)


        #DeepHQ
        TINY = 1e-5
        self.QV = nn.Parameter(torch.clamp(torch.ones(8, self.M), min=0.0) + TINY)
        self.IQV = nn.Parameter(torch.clamp(torch.ones(8, self.M), min=0.0) + TINY)
        self.gamma = nn.Parameter(torch.clamp(torch.ones(8, self.M), min=0.0))
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    def get_masks(self, importance_map):
        mask = torch.zeros_like(importance_map)
        masks = []
        for idx, gamma_q in enumerate(self.gamma):
            adjusted_importance_map = importance_map ** gamma_q.view(1, -1, 1, 1)
            if self.training:
                u = torch.rand_like(adjusted_importance_map)
                samples = u + (adjusted_importance_map - 0.5)
                rounded = ste_round(samples)
            else: # for test
                rounded = torch.round(adjusted_importance_map)
            mask = mask + (1 - mask) * rounded
            # mask = rounded
            masks.append(mask.clone())
        return masks
    
    # def get_masks(self, latent_means, latent_scales):
    #     importance_map = self.importance_map(torch.cat([latent_means, latent_scales], dim=1))
    #     mask = torch.zeros_like(latent_means)
    #     mask = torch.zeros_like(importance_map)
    #     masks = []
    #     for idx, gamma_q in enumerate(self.gamma):
    #         adjusted_importance_map = importance_map ** gamma_q.view(1, -1, 1, 1)
    #         if self.training:
    #             u = torch.rand_like(adjusted_importance_map)
    #             samples = u + (adjusted_importance_map - 0.5)
    #             rounded = ste_round(samples)
    #         else: # for test
    #             rounded = torch.round(adjusted_importance_map)
    #         # mask = mask + (1 - mask) * rounded
    #         mask = rounded
    #         masks.append(mask.clone())
    #     return masks

    def _register_nan_inf_hook(self, name, tensor):
        if tensor.requires_grad:
            def hook_fn(grad):
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print(f"[Gradient Anomaly Detected] in {name}")
                    print(f"  min: {grad.min().item()}, max: {grad.max().item()}, mean: {grad.mean().item()}")
                    print(f"  grad shape: {grad.shape}")
            tensor.register_hook(hook_fn)


    def forward(self, x):
        outnet_list = []

        # with torch.no_grad():
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        importance_map = self.importance_map(z_hat)
        y_slices = y.chunk(self.num_slices, 1)
        # masks = self.get_masks(latent_means, latent_scales)
        masks = self.get_masks(importance_map)

        for q_idx in range(8):
            # with torch.no_grad():
            y_likelihood = []
            
            slice_index = 0
            y_slice = y_slices[0]
            # support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            support_slices = []
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            # mu_list.append(mu)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            # self._register_nan_inf_hook("scale_support", scale_support)

            scale = self.cc_scale_transforms[slice_index](scale_support)
            # self._register_nan_inf_hook("scale", scale)

            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            # scale = torch.clamp(scale, min=1e-6)
            # scale = np.maximum(scale, SCALE_LOWER_BOUND)
            
            # scale_list.append(scale)
            
            #DeepHQ Phase 2
            y_slice_q = y_slice / self.QV[q_idx].view(1, -1, 1, 1)
            scale_q = scale / self.QV[q_idx].view(1, -1, 1, 1)
            mu_q = mu / self.QV[q_idx].view(1, -1, 1, 1)
                    
            _, y_slice_likelihood = self.gaussian_conditional(y_slice_q, scale_q, mu_q)
            y_likelihood.append(y_slice_likelihood)
            
            # y_hat_slice = ste_round(y_slice_q - mu_q) + mu_q
            y_hat_slice = ste_round(y_slice_q)

            # mask = masks[q_idx]
            # print(y_slice_q[mask.bool()])
            # print(y_hat_slice[mask.bool()])
            # asdf

            y_hat_slice = y_hat_slice * self.IQV[q_idx].view(1, -1, 1, 1)
            ##################end no grad ################################
            
            #DeepHQ Phase 3
            mask = masks[q_idx]
            combined_y_hat = mask * y_hat_slice + (1 - mask) * (
                    mu_q * self.IQV[q_idx].view(1, -1, 1, 1))

            # if self.training:
            #     lrp_support = torch.cat([mean_support + torch.randn(mean_support.size()).cuda().mul(scale_support), y_hat_slice], dim=1)
            # else:
            lrp_support = torch.cat([mean_support, combined_y_hat], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            combined_y_hat += lrp
            y_hat = combined_y_hat
            
            # means = torch.cat(mu_list, dim=1)
            # scales = torch.cat(scale_list, dim=1)
            y_likelihoods = torch.cat(y_likelihood, dim=1)
            
            # #for easy training
            # y_likelihoods[mask == 0] = 1.0

            x_hat = self.g_s(y_hat)
            outnet = {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "mask": mask,
            }
            outnet_list.append(outnet)
        return outnet_list

    def load_state_dict(self, state_dict, strict=True):
        def strip_module_prefix(state_dict):
            return {
                k.replace("module.", "", 1) if k.startswith("module.") else k: v
                for k, v in state_dict.items()
            }
        state_dict = strip_module_prefix(state_dict)
        # for k in state_dict.keys():
        #     print(k)
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    # def compress(self, x):
    #     y = self.g_a(x)
    #     y_shape = y.shape[2:]

    #     z = self.h_a(y)
    #     z_strings = self.entropy_bottleneck.compress(z)
    #     z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    #     latent_scales = self.h_scale_s(z_hat)
    #     latent_means = self.h_mean_s(z_hat)

    #     y_slices = y.chunk(self.num_slices, 1)
    #     y_hat_slices = []
    #     y_scales = []
    #     y_means = []

    #     cdf = self.gaussian_conditional.quantized_cdf.tolist()
    #     cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
    #     offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

    #     gamma = torch.relu(self.raw_gamma)
    #     mask = torch.zeros_like(y)

    #     results = []
    #     for q_idx in range(6):
    #         encoder = BufferedRansEncoder()
    #         symbols_list = []
    #         indexes_list = []
    #         y_strings = []
            
    #         slice_index = 0
    #         y_slice = y_slices[0]
    #         support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

    #         mean_support = torch.cat([latent_means] + support_slices, dim=1)
    #         mean_support = self.atten_mean[slice_index](mean_support)
    #         mu = self.cc_mean_transforms[slice_index](mean_support)
    #         mu = mu[:, :, :y_shape[0], :y_shape[1]]

    #         scale_support = torch.cat([latent_scales] + support_slices, dim=1)
    #         scale_support = self.atten_scale[slice_index](scale_support)
    #         scale = self.cc_scale_transforms[slice_index](scale_support)
    #         scale = scale[:, :, :y_shape[0], :y_shape[1]]

    #         #DeepHQ
    #         y_slice = y_slice / self.QV[q_idx].view(1, -1, 1, 1)
    #         scale = scale / self.QV[q_idx].view(1, -1, 1, 1)
    #         mu = mu / self.QV[q_idx].view(1, -1, 1, 1)

    #         #DeepHQ Phase 3
    #         mask = self.get_mask(q_idx, gamma, latent_means, latent_scales, mask)

    #         # Just for quick evaluation for training phase 3. Use "eval_DeepHQ.py" for the exact evaluation of the scalable coding functions.
    #         # zero mean, zero scale for zero values do not yeield bitrate consumption.
    #         y_slice = y_slice * mask
    #         mu =  mu * mask
    #         scale =  scale * mask

    #         index = self.gaussian_conditional.build_indexes(scale)
    #         y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
    #         y_hat_slice = y_q_slice + mu

    #         symbols_list.extend(y_q_slice.reshape(-1).tolist())
    #         indexes_list.extend(index.reshape(-1).tolist())


    #         lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
    #         lrp = self.lrp_transforms[slice_index](lrp_support)
    #         lrp = 0.5 * torch.tanh(lrp)
    #         y_hat_slice += lrp

    #         y_hat_slices.append(y_hat_slice)
    #         y_scales.append(scale)
    #         y_means.append(mu)

    #         encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
    #         y_string = encoder.flush()
    #         y_strings.append(y_string)
    #         results.append({"strings": [y_strings, z_strings], "shape": z.size()[-2:]})
    #     return results

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        slice_index = 0
        y_slice = y_slices[0]
        support_slices = []

        mean_support = torch.cat([latent_means] + support_slices, dim=1)
        mean_support = self.atten_mean[slice_index](mean_support)
        mu = self.cc_mean_transforms[slice_index](mean_support)
        mu = mu[:, :, :y_shape[0], :y_shape[1]]

        scale_support = torch.cat([latent_scales] + support_slices, dim=1)
        scale_support = self.atten_scale[slice_index](scale_support)
        scale = self.cc_scale_transforms[slice_index](scale_support)
        scale = scale[:, :, :y_shape[0], :y_shape[1]]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        gamma = self.gamma

        results = []
        # masks = self.get_masks(latent_means, latent_scales)
        importance_map = self.importance_map(z_hat)
        masks = self.get_masks(importance_map)

        # for mask in (masks):
        #     print(mask.mean())
        # asdf
        
        for q_idx in range(8):
            encoder = BufferedRansEncoder()
            symbols_list = []
            indexes_list = []
            y_strings = []
            

            #DeepHQ
            y_slice_q = y_slice / self.QV[q_idx].view(1, -1, 1, 1)
            scale_q = scale / self.QV[q_idx].view(1, -1, 1, 1)
            mu_q = mu / self.QV[q_idx].view(1, -1, 1, 1)

            #DeepHQ Phase 3
            mask = masks[q_idx]
            mask_bool = mask.bool()

            index = self.gaussian_conditional.build_indexes(scale_q)
            y_q_slice = self.gaussian_conditional.quantize(y_slice_q, "symbols", mu_q)


            y_slice_q = y_slice_q[mask_bool]
            mu_q =  mu_q[mask_bool]
            scale_q =  scale_q[mask_bool]
            # y_slice = y_slice * mask
            # mu_q =  mu_q * mask
            # scale_q =  scale_q * mask

            scale = torch.clamp(scale, min=0.11)
  


            index = self.gaussian_conditional.build_indexes(scale_q)
            
            y_q_slice = self.gaussian_conditional.quantize(y_slice_q, "symbols", mu_q)
            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())
            encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
            y_string = encoder.flush()
            # print(cdf_lengths)
            # print(len(cdf_lengths))
            # asdf
            
            # alternative
            # y_string = self.gaussian_conditional.compress(y_slice_q.unsqueeze(0), index.unsqueeze(0))
            

            y_strings.append(y_string)
            results.append({"strings": [y_strings, z_strings], "shape": z.size()[-2:]})
        return results

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)
    
    def get_DELTA(self, quant_layer_index_l):
        if quant_layer_index_l - (quant_layer_index_l // 1) == 0: # integer level
            quant_layer_index_l = int(quant_layer_index_l)
            DELTA_l = self.QV[quant_layer_index_l]
            INVERSE_DELTA_l = self.IQV[quant_layer_index_l]
        else: # in-between
            lower_layer = quant_layer_index_l // 1
            upper_layer = lower_layer + 1
            decimal_part = quant_layer_index_l - lower_layer

            low_DELTA_l = self.QV[int(lower_layer)]
            low_INVERSE_DELTA_l = self.IQV[int(lower_layer)]
            high_DELTA_l = self.QV[int(upper_layer)]
            high_INVERSE_DELTA_l = self.IQV[int(upper_layer)]

            # DELTA_l = low_DELTA_l ** (1 - decimal_part) * high_DELTA_l ** decimal_part
            # INVERSE_DELTA_l = low_INVERSE_DELTA_l ** (1 - decimal_part) * high_INVERSE_DELTA_l ** decimal_part
            DELTA_l = (1 - decimal_part) * low_DELTA_l + decimal_part * high_DELTA_l
            INVERSE_DELTA_l = (1 - decimal_part) * low_INVERSE_DELTA_l + decimal_part * high_INVERSE_DELTA_l
        return DELTA_l, INVERSE_DELTA_l

    def decompress(self, out_enc_list):
        out_dec_list = []
        gamma=self.gamma

        
        out_dec_list = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        shape = out_enc_list[0]["shape"]
        z_hat = self.entropy_bottleneck.decompress(out_enc_list[0]["strings"][1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        
        slice_index = 0
        # support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
        support_slices = []
        mean_support = torch.cat([latent_means] + support_slices, dim=1)
        mean_support = self.atten_mean[slice_index](mean_support)
        mu = self.cc_mean_transforms[slice_index](mean_support)
        mu = mu[:, :, :y_shape[0], :y_shape[1]]

        scale_support = torch.cat([latent_scales] + support_slices, dim=1)
        scale_support = self.atten_scale[slice_index](scale_support)
        scale = self.cc_scale_transforms[slice_index](scale_support)
        scale = scale[:, :, :y_shape[0], :y_shape[1]]

        # masks = self.get_masks(latent_means, latent_scales)
        importance_map = self.importance_map(z_hat)
        masks = self.get_masks(importance_map)
        for q_idx, out_enc in enumerate(out_enc_list):
            y_hat_slice = torch.zeros_like(scale)

            strings = out_enc["strings"]
            y_string = strings[0][0]
            y_hat_slices = []
            decoder = RansDecoder()
            decoder.set_stream(y_string)       
            
            #DeepHQ
            scale_q = scale / self.QV[q_idx].view(1, -1, 1, 1)
            mu_q = mu / self.QV[q_idx].view(1, -1, 1, 1)
            #DeepHQ Phase 3
            mask = masks[q_idx]
            # mask,_,_ = self.get_mask(q_idx, gamma, mean_support, scale_support, mask)

            mask_bool = mask.bool()
            mu_q_selected =  mu_q[mask_bool]
            scale_q_selected =  scale_q[mask_bool]

            index = self.gaussian_conditional.build_indexes(scale_q_selected)
            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            # rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            rv = torch.Tensor(rv).reshape(scale_q_selected.shape)
            y_hat_slice_selected = self.gaussian_conditional.dequantize(rv, mu_q_selected)

            
            # print(y_hat_slice_selected)
            # asdf
            y_hat_slice[mask_bool] = y_hat_slice_selected
            y_hat_slice = y_hat_slice * self.IQV[q_idx].view(1, -1, 1, 1)

            y_hat_slice = y_hat_slice + (1 - mask) * (
                    mu_q * self.IQV[q_idx].view(1, -1, 1, 1))          

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            # y_hat_slices.append(y_hat_slice)
            # y_hat = torch.cat(y_hat_slices, dim=1)

            x_hat = self.g_s(y_hat_slice).clamp_(0, 1)
            out_dec_list.append({"x_hat": x_hat})

        return out_dec_list
    
    # def decompress(self, out_enc_list):
    #     out_dec_list = []
        
    #     gamma = self.gamma
    #     mask = torch.zeros_like(y)

    #     for out_enc in out_enc_list:
    #         strings = out_enc["strings"]
    #         shape = out_enc["shape"]

    #         z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
    #         latent_scales = self.h_scale_s(z_hat)
    #         latent_means = self.h_mean_s(z_hat)

    #         y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

    #         y_string = strings[0][0]

    #         y_hat_slices = []
    #         cdf = self.gaussian_conditional.quantized_cdf.tolist()
    #         cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
    #         offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

    #         decoder = RansDecoder()
    #         decoder.set_stream(y_string)

    #         for q_idx in range(6):
    #             slice_index = 0
    #             support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
    #             mean_support = torch.cat([latent_means] + support_slices, dim=1)
    #             mean_support = self.atten_mean[slice_index](mean_support)
    #             mu = self.cc_mean_transforms[slice_index](mean_support)
    #             mu = mu[:, :, :y_shape[0], :y_shape[1]]

    #             scale_support = torch.cat([latent_scales] + support_slices, dim=1)
    #             scale_support = self.atten_scale[slice_index](scale_support)
    #             scale = self.cc_scale_transforms[slice_index](scale_support)
    #             scale = scale[:, :, :y_shape[0], :y_shape[1]]

    #             #DeepHQ
    #             scale = scale / self.QV[q_idx].view(1, -1, 1, 1)
    #             mu = mu / self.QV[q_idx].view(1, -1, 1, 1)
    #             reshaped_y_hat_slice = np.zeros_like(pred_sigma)

    #             #DeepHQ Phase 3
    #             mask,_,_ = self.get_mask(q_idx, gamma, latent_means, latent_scales, mask)
    #             scale = scale[mask]
    #             mu = mu[mask]

    #             index = self.gaussian_conditional.build_indexes(scale)

    #             rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
    #             rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
    #             y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)
    #             y_hat_slice[mask] = y_hat_slice[0]
                
    #             combined_y_hat = y_hat_slice + (1 - mask) * (
    #                 mu * self.IQV[q_idx].view(1, -1, 1, 1))
                
    #             lrp_support = torch.cat([mean_support, combined_y_hat], dim=1)
    #             lrp = self.lrp_transforms[slice_index](lrp_support)
    #             lrp = 0.5 * torch.tanh(lrp)
    #             y_hat_slice += lrp

    #             y_hat_slices.append(y_hat_slice)

    #         y_hat = torch.cat(y_hat_slices, dim=1)
    #         x_hat = self.g_s(y_hat).clamp_(0, 1)
    #         out_dec_list.append({"x_hat": x_hat})

    #     return out_dec_list
    
    def dhq_compress(self, x, continuous_compression=False):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])
        results = {}
        results["z_string"] = z_string
        

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        slice_index = 0
        y_slice = y_slices[0]
        support_slices = []

        mean_support = torch.cat([latent_means] + support_slices, dim=1)
        mean_support = self.atten_mean[slice_index](mean_support)
        mu = self.cc_mean_transforms[slice_index](mean_support)
        mu = mu[:, :, :y_shape[0], :y_shape[1]]

        scale_support = torch.cat([latent_scales] + support_slices, dim=1)
        scale_support = self.atten_scale[slice_index](scale_support)
        scale = self.cc_scale_transforms[slice_index](scale_support)
        scale = scale[:, :, :y_shape[0], :y_shape[1]]


        y_star = y - mu
        # NUM_TOTAL_SYMBOLS_K = math.ceil(np.max(y_star / self.QV[0].view(1,-1,1,1))) * 2 + 1
        NUM_TOTAL_SYMBOLS_K = math.ceil((y_star / self.QV[0].view(1, -1, 1, 1)).max().item()) * 2 + 1
        Q_LEVEL = 8
        n_partitions = 23
        threshold_T = 0.3

        mask = torch.zeros_like(y)
        strings_whole = []
        # masks = self.get_masks(latent_means, latent_scales)
        importance_map = self.importance_map(z_hat)
        masks = self.get_masks(importance_map)
        # for mask in (masks):
        #     print(mask.mean())
        # asdf
        for quant_layer_index_l in range(Q_LEVEL):
            print(f"quant. layer: {quant_layer_index_l}")
            DELTA_l = self.QV[quant_layer_index_l]
            INVERSE_DELTA_l = self.IQV[quant_layer_index_l]
            
            ############### encode y_hat ####################################
            mask = masks[quant_layer_index_l]
            if quant_layer_index_l == 0:
                LB = DELTA_l * (NUM_TOTAL_SYMBOLS_K/2) * -1.0
                # LB = np.expand_dims(np.expand_dims(LB, axis=-1), axis=-1)
                LB = LB.unsqueeze(-1).unsqueeze(-1)
                # LB = np.tile(LB, (1, y.shape[1], y.shape[2]))  # tiling by (H,W) to make (C,H,W)
                LB = LB.repeat(1, y.shape[2], y.shape[3])

                UB = DELTA_l * (NUM_TOTAL_SYMBOLS_K/2)
                # UB = np.expand_dims(np.expand_dims(UB, axis=-1), axis=-1)
                UB = UB.unsqueeze(-1).unsqueeze(-1)
                # UB = np.tile(UB, (1, y.shape[1], y.shape[2]))  # tiling by (H,W) to make (C,H,W)
                UB = UB.repeat(1, y.shape[2], y.shape[3])
            
            # print(LB.shape)
            boundaries, level_indexes, LB_new, UB_new = get_boundary_list(mask[0], y_star[0, :, :, :], LB,
                                                                          UB, DELTA_l,
                                                                          NUM_TOTAL_SYMBOLS=NUM_TOTAL_SYMBOLS_K,
                                                                          interval_lower_bound=threshold_T)
            
            # update ranges for the next layer
            LB = LB_new
            UB = UB_new
            
            f_selected_pred_sigma = scale[mask.bool()]
            cdf_list = get_cdf_list(boundaries, f_selected_pred_sigma)
            # cdf_list = np.transpose(cdf_list, (1, 0)) # N_mask, S
            cdf_list = cdf_list.permute(1, 0)  # N_mask x S

            cdf_list = cdf_list.cpu().numpy().astype(np.float32)
            cdf_list_temp = np.ones(cdf_list.shape, dtype=np.float32)
            cdf_list_temp[:, :] = cdf_list[:, :]
            cdf_list = cdf_list_temp #Unsolved issue. This copy process is necessary for working properly.
            cdf_list = torch.from_numpy(cdf_list)
            
            strings = []


            ###########################
            if continuous_compression and quant_layer_index_l != 0: #this code is performed only for the fined-grained progressive coding (when 'is_latentordering' option is enabled)
                criteria_data = scale[mask.bool()].reshape(-1)
                sorted_index = torch.argsort(criteria_data, descending=True).tolist()


                # first_partition_size = int(len(criteria_data) * 0.3)
                normal_partition_size = int(len(criteria_data) / n_partitions)
                encoded_size = 0
                for partition_idx in range(n_partitions):
                    print(f"encoding... q_layer:{quant_layer_index_l}, partition no.:{partition_idx}")
                    #######################################################
                    # Determine partition size
                    if partition_idx == n_partitions - 1:
                        partition_size = len(criteria_data) - normal_partition_size * (n_partitions - 1)
                    else:
                        partition_size = normal_partition_size

                    # Get sorted partition indexes
                    partition_indexes = sorted_index[encoded_size:encoded_size + partition_size]
                    partition_indexes.sort()
                    encoded_size += partition_size

                    # Select partitioned CDF and level indexes
                    partition_cdf_list = torch.tensor(cdf_list[partition_indexes], dtype=torch.float32)
                    partition_level_indexes = torch.tensor(level_indexes[partition_indexes], dtype=torch.int32)

                    # Encode using torchac
                    partition_level_indexes = partition_level_indexes.to(dtype=torch.int16).cpu()
                    bytestream = torchac.encode_float_cdf(
                        partition_cdf_list, partition_level_indexes,
                        check_input_bounds=True, needs_normalization=True
                    )
                    strings.append(bytestream)
            else:
                level_indexes = level_indexes.to(dtype=torch.int16).cpu()
                # print(level_indexes.to(dtype=torch.int64))
                # print(level_indexes)
                # print(cdf_list.shape)
                # asdfasfd

                byte_stream = torchac.encode_float_cdf(cdf_list, level_indexes, check_input_bounds=True,
                                                       needs_normalization=True)
                
                gathered1 = torch.gather(cdf_list, dim=1, index=(level_indexes + 1).to(dtype=torch.int64).unsqueeze(1))  # (N, 1)
                gathered1 = gathered1.squeeze(1) 

                gathered2 = torch.gather(cdf_list, dim=1, index=level_indexes.to(dtype=torch.int64).unsqueeze(1))  # (N, 1)
                gathered2 = gathered2.squeeze(1) 

                likelihoods = gathered1 - gathered2
                # bits = torch.sum(-torch.log(likelihoods + 1e-5) / torch.log(torch.tensor(2.0)))
                # print(f"estimated bits:{bits}")
                # print(f"actual bits:{len(byte_stream) * 8}")
                strings.append(byte_stream)
            
            strings_whole.append(strings)
        results["y_string"] = strings_whole
        return results, NUM_TOTAL_SYMBOLS_K
    
    def dhq_decompress(self, enc_results, x_shape, NUM_TOTAL_SYMBOLS_K, continuous_compression=False):
        y_h = x_shape[2] // 16
        y_w = x_shape[3] // 16
        z_h = y_h // 4
        z_w = y_w // 4

        z_string = enc_results["z_string"]
        z_hat = self.entropy_bottleneck.decompress(z_string, (z_h, z_w))
        results = {}
        results["z_string"] = z_string
        

        y_hat_slices = []
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        y_string = enc_results["y_string"]
        total_string_len = len(z_string)

        ################
        slice_index = 0
        support_slices = []
        mean_support = torch.cat([latent_means] + support_slices, dim=1)
        mean_support = self.atten_mean[slice_index](mean_support)
        mu = self.cc_mean_transforms[slice_index](mean_support)
        mu = mu[:, :, :y_h, :y_w]

        scale_support = torch.cat([latent_scales] + support_slices, dim=1)
        scale_support = self.atten_scale[slice_index](scale_support)
        scale = self.cc_scale_transforms[slice_index](scale_support)
        scale = scale[:, :, :y_h, :y_w]
        
        Q_LEVEL = 8
        n_partitions = 23
        threshold_T = 0.3
        y_breve_star = torch.zeros_like(scale)

        out_dec_list_whole = []
        # masks = self.get_masks(latent_means, latent_scales)
        importance_map = self.importance_map(z_hat)
        masks = self.get_masks(importance_map)
        for quant_layer_index_l in range(Q_LEVEL):
            print(f"quality_level: {quant_layer_index_l}")
            DELTA_l =  self.QV[quant_layer_index_l]
            INVERSE_DELTA_l = self.IQV[quant_layer_index_l]


            y_string_q = y_string[quant_layer_index_l]

            if quant_layer_index_l == 0:
                LB = DELTA_l * (NUM_TOTAL_SYMBOLS_K/2) * -1.0
                LB = LB.unsqueeze(-1).unsqueeze(-1)
                LB = LB.repeat(1, y_h, y_w)

                UB = DELTA_l * (NUM_TOTAL_SYMBOLS_K/2)
                UB = UB.unsqueeze(-1).unsqueeze(-1)
                UB = UB.repeat(1, y_h, y_w)
            
            mask = masks[quant_layer_index_l]
            boundaries, _, _, _ = get_boundary_list(mask[0], None, LB, UB, DELTA_l,
                                                    NUM_TOTAL_SYMBOLS=NUM_TOTAL_SYMBOLS_K,
                                                    interval_lower_bound=threshold_T)
            
            f_selected_pred_sigma = scale[mask.bool()]
            cdf_list = get_cdf_list(boundaries, f_selected_pred_sigma)
            # cdf_list = np.transpose(cdf_list, (1, 0)) # N_mask, S
            cdf_list = cdf_list.permute(1, 0)  # N_mask x S

            cdf_list = cdf_list.cpu().numpy().astype(np.float32)
            cdf_list_temp = np.ones(cdf_list.shape, dtype=np.float32)
            cdf_list_temp[:, :] = cdf_list[:, :]
            cdf_list = cdf_list_temp #Unsolved issue. This copy process is necessary for working properly.
            cdf_list = torch.from_numpy(cdf_list)
            


            ###########################
            out_dec_list = []
            if continuous_compression and quant_layer_index_l != 0: #this code is performed only for the fined-grained progressive coding (when 'is_latentordering' option is enabled)
                # Flatten and sort criteria values
                criteria_data = scale[mask.bool()].reshape(-1)
                sorted_index = torch.argsort(criteria_data, descending=True).tolist()
                normal_partition_size = len(criteria_data) // n_partitions

                decoded_size = 0

                for partition_idx in range(n_partitions):
                    print(f"decoding... q_layer:{quant_layer_index_l}, partition no.:{partition_idx}")
                    # Determine partition size
                    if partition_idx == n_partitions - 1:
                        partition_size = len(criteria_data) - normal_partition_size * (n_partitions - 1)
                    else:
                        partition_size = normal_partition_size

                    # Get sorted indexes for the current partition
                    partition_indexes = sorted_index[decoded_size:decoded_size + partition_size]
                    partition_indexes.sort()
                    decoded_size += partition_size

                    # Get the corresponding CDF and byte stream
                    parition_cdf_list = torch.tensor(cdf_list[partition_indexes], dtype=torch.float32)
                    byte_stream = y_string_q[partition_idx]

                    # Decode the byte stream into quantized indexes (placeholder below)
                    parition_level_indexes = torchac.decode_float_cdf(parition_cdf_list, byte_stream)

                    # Get quantization boundaries for this partition
                    partition_boundaries = torch.tensor(boundaries[:, partition_indexes], dtype=torch.float32)

                    # Build binary mask for selected partition
                    
                    # Create a flat boolean tensor of the same number of True values as mask.sum()
                    partition_mask_true_subset = torch.zeros(mask.bool().sum(), dtype=torch.bool, device=mask.device)
                    partition_mask_true_subset[partition_indexes] = True

                    # Create final mask of the same shape as `mask`, all False initially
                    partition_mask = torch.zeros_like(mask, dtype=torch.bool)

                    # Fill in True values where original mask was True, using the partitioned subset
                    partition_mask[mask.bool()] = partition_mask_true_subset


                    # Perform reconstruction for current partition
                    partition_reconstructions, LB, UB = get_reconstructions(
                        partition_boundaries, parition_level_indexes, LB, UB, partition_mask[0]
                    )
                    y_breve_star[partition_mask] = partition_reconstructions

                    # Update scale factors (quantization step size)
                    cont_DELTA_l, cont_INVERSE_DELTA_l = self.get_DELTA(
                        quant_layer_index_l - 1 + ((partition_idx + 1) / n_partitions)
                    )

                    y_breve_final = ((y_breve_star + mu) / cont_DELTA_l.view(1, -1, 1, 1) * cont_INVERSE_DELTA_l.view(1, -1, 1, 1))
                    # y_breve_final = ((y_breve_star + mu) / DELTA_l.view(1, -1, 1, 1) * INVERSE_DELTA_l.view(1, -1, 1, 1))

                    

                    lrp_support = torch.cat([mean_support, y_breve_final], dim=1)
                    lrp = self.lrp_transforms[slice_index](lrp_support)
                    lrp = 0.5 * torch.tanh(lrp)
                    y_breve_final += lrp

                    # Reconstruct image (replace with actual decoder model if needed)
                    x_hat = self.g_s(y_breve_final).clamp_(0, 1)  # simulated output for debugging
                    out_dec_list.append(x_hat)

            else:
                byte_stream = y_string_q[0]
                level_indexes = torchac.decode_float_cdf(cdf_list, byte_stream)
                # codes for mapping index to reconstructions value, b_lo_new, and b_up_new
                reconstructions, LB_new, UB_new = get_reconstructions(boundaries, level_indexes, LB, UB, mask[0])

                #update range
                LB = LB_new
                UB = UB_new

                y_breve_star[mask.bool()] = reconstructions
                y_breve_final = ((y_breve_star + mu) / DELTA_l.view(1, -1, 1, 1) * INVERSE_DELTA_l.view(1, -1, 1, 1))
                
                
                
                lrp_support = torch.cat([mean_support, y_breve_final], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_breve_final += lrp
                x_hat = self.g_s(y_breve_final).clamp_(0, 1)
                out_dec_list.append(x_hat)

            out_dec_list_whole.append(out_dec_list)
        return out_dec_list_whole





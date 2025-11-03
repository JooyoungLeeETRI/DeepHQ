from __future__ import print_function

import os
import math
import json
import numpy as np
import torch
from PIL import Image
from datetime import datetime
from numba import jit

import torch

# @jit(nopython=True, parallel=True)
# def get_boundary_base(f_selected_LB, f_selected_UB, NUM_TOTAL_SYMBOLS_K):
#     f_selected_center_v = (f_selected_LB + f_selected_UB) / 2 # lower-level recon
#     baseints = - (NUM_TOTAL_SYMBOLS_K - 1) / 2 # -(K-1)_ / 2
#     boundary_base = np.linspace(baseints, baseints + NUM_TOTAL_SYMBOLS_K, NUM_TOTAL_SYMBOLS_K + 1) # range of k
#     boundary_base -= 0.5 #(k-0.5)
#     boundary_base = np.expand_dims(boundary_base, axis=1)  # K x 1 dim
#     # boundary_base = boundary_base.repeat(f_selected_LB.shape[0]).reshape((-1, f_selected_LB.shape[0])) # Nsym x N_mask (N_mask: total number of selected elements)
#     boundary_base = np.tile(boundary_base, (1, f_selected_LB.shape[0])) # Nsym x N_mask (N_mask: total number of selected elements)
#     return f_selected_center_v, boundary_base

def get_boundary_base(f_selected_LB, f_selected_UB, NUM_TOTAL_SYMBOLS_K):
    # f_selected_LB: Tensor of shape [N_mask]
    # f_selected_UB: Tensor of shape [N_mask]
    # returns:
    #   - f_selected_center_v: Tensor of shape [N_mask]
    #   - boundary_base: Tensor of shape [K+1, N_mask]

    f_selected_center_v = (f_selected_LB + f_selected_UB) / 2  # Tensor [N_mask]
    baseints = - (NUM_TOTAL_SYMBOLS_K - 1) / 2

    # Create [K+1] values from baseints to baseints + K, then subtract 0.5 for (k - 0.5)
    boundary_base = torch.linspace(
        baseints, baseints + NUM_TOTAL_SYMBOLS_K, NUM_TOTAL_SYMBOLS_K + 1,
        device=f_selected_LB.device, dtype=f_selected_LB.dtype
    ) - 0.5  # shape: [K+1]

    # Expand to shape [K+1, N_mask]
    boundary_base = boundary_base.unsqueeze(1).expand(-1, f_selected_LB.shape[0])

    return f_selected_center_v, boundary_base


# @jit(nopython=True, parallel=True)
# def get_boundaries(f_selected_y_star, boundaries, f_selected_center_v, f_selected_LB, f_selected_UB, f_selected_DELTA_l, T):
#     #input boundaries are "k-0.5" in the paper
#     r = ((f_selected_center_v - f_selected_LB - 0.5 * f_selected_DELTA_l) % f_selected_DELTA_l) / f_selected_DELTA_l # r: interval ratio compared to the normal intervals
#     ddot_f_selected_DELTA_l = (f_selected_center_v - f_selected_LB) / (
#             np.maximum((f_selected_center_v - f_selected_LB - 0.5 * f_selected_DELTA_l),
#                        0) // f_selected_DELTA_l + 0.5) # \ddot{delta} in the paper. "(f_selected_center - f_selected_b_lo_previous)" is (UB-LB)/2 in the paper
#     normal_boundaries = boundaries * f_selected_DELTA_l + f_selected_center_v  # Nsym x N_mask
#     ddot_boundaries = boundaries * ddot_f_selected_DELTA_l + f_selected_center_v  # Nsym x N_mask, \ddot{b} in the paper.
#     r = r.repeat(boundaries.shape[0]).reshape((r.shape[0], boundaries.shape[0])) # N_mask x Nsym
#     r = np.transpose(r)  # Nsym x N_mask
#     boundaries = np.where(r < T, ddot_boundaries, normal_boundaries)

#     #for index calculation
#     if f_selected_y_star is not None:
#         a = (boundaries < f_selected_y_star)[:boundaries.shape[0] - 1, :]
#         b = (boundaries > f_selected_y_star)[1:, :]
#     else:
#         a = np.zeros_like(boundaries,dtype=np.bool_)[:boundaries.shape[0] - 1, :]
#         b = np.zeros_like(boundaries,dtype=np.bool_)[:boundaries.shape[0] - 1, :]

#     boundaries = np.maximum(boundaries, f_selected_LB)
#     boundaries = np.minimum(boundaries, f_selected_UB)
#     return boundaries, a, b

def get_boundaries(f_selected_y_star, boundaries, f_selected_center_v, f_selected_LB, f_selected_UB, f_selected_DELTA_l, T):
    """
    All inputs are torch.Tensor
    Shapes:
        boundaries: [Nsym, N_mask]
        f_selected_center_v, f_selected_LB, f_selected_UB, f_selected_DELTA_l: [N_mask]
        f_selected_y_star: [N_mask] or None
    """

    # r: interval ratio
    r = ((f_selected_center_v - f_selected_LB - 0.5 * f_selected_DELTA_l) % f_selected_DELTA_l) / f_selected_DELTA_l

    # \ddot{delta}
    offset = f_selected_center_v - f_selected_LB - 0.5 * f_selected_DELTA_l
    offset_clipped = torch.clamp(offset, min=0.0)
    denom = (offset_clipped // f_selected_DELTA_l) + 0.5
    ddot_f_selected_DELTA_l = (f_selected_center_v - f_selected_LB) / denom

    # Expand to shape [Nsym, N_mask]
    f_selected_DELTA_l_exp = f_selected_DELTA_l.unsqueeze(0).expand(boundaries.shape)
    ddot_f_selected_DELTA_l_exp = ddot_f_selected_DELTA_l.unsqueeze(0).expand(boundaries.shape)
    f_selected_center_v_exp = f_selected_center_v.unsqueeze(0).expand(boundaries.shape)

    # Compute normal and ddot boundaries
    normal_boundaries = boundaries * f_selected_DELTA_l_exp + f_selected_center_v_exp
    ddot_boundaries = boundaries * ddot_f_selected_DELTA_l_exp + f_selected_center_v_exp

    # Expand r to shape [Nsym, N_mask]
    r_exp = r.unsqueeze(0).expand(boundaries.shape)

    # Adaptive boundary selection
    boundaries = torch.where(r_exp < T, ddot_boundaries, normal_boundaries)

    # Index masks for interval lookup
    if f_selected_y_star is not None:
        f_selected_y_star_exp = f_selected_y_star.unsqueeze(0).expand(boundaries.shape)
        a = (boundaries < f_selected_y_star_exp)[:-1, :]
        b = (boundaries > f_selected_y_star_exp)[1:, :]
    else:
        a = torch.zeros_like(boundaries[:-1, :], dtype=torch.bool)
        b = torch.zeros_like(boundaries[:-1, :], dtype=torch.bool)

    # Clip to [LB, UB]
    f_selected_LB_exp = f_selected_LB.unsqueeze(0).expand(boundaries.shape)
    f_selected_UB_exp = f_selected_UB.unsqueeze(0).expand(boundaries.shape)
    boundaries = torch.maximum(boundaries, f_selected_LB_exp)
    boundaries = torch.minimum(boundaries, f_selected_UB_exp)

    return boundaries, a, b



@jit(nopython=True, parallel=True)
def get_new_upper_lower_bounds(boundaries, indexes):
    temp_indexes = np.expand_dims(indexes, axis=0)
    f_selected_b_lo_new = np.take_along_axis(boundaries, temp_indexes, axis=0)[0]
    f_selected_b_up_new = np.take_along_axis(boundaries, temp_indexes + 1, axis=0)[0]
    return f_selected_b_lo_new, f_selected_b_up_new

# def get_boundary_list(mask, y_star, LB, UB, DELTA_l, NUM_TOTAL_SYMBOLS=21, interval_lower_bound=0.3):
#     # LB, UB: CHW
#     # mask: CHW
#     # DELTA_l: C

#     indexes = None
#     LB_new = None
#     UB_new = None
#     mask=mask.bool()


#     # flatten mask, y_star, LB, UB
#     if y_star is not None:
#         f_selected_y_star = y_star[mask]
#     else:
#         f_selected_y_star = None

#     f_selected_LB = LB[mask]  # shape: N_selected
#     f_selected_UB = UB[mask]

#     # compute center and boundaries
#     f_selected_center_v, boundary_base = get_boundary_base(f_selected_LB, f_selected_UB, NUM_TOTAL_SYMBOLS)

#     # expand and tile DELTA_l: from (C,) to (C, H, W)
#     C, H, W = LB.shape
#     # DELTA_l = np.expand_dims(DELTA_l, axis=(1, 2))  # C x 1 x 1
#     DELTA_l = DELTA_l.view(-1, 1, 1)
#     # DELTA_l = np.tile(DELTA_l, (1, H, W))           # C x H x W
#     DELTA_l = DELTA_l.view(C, 1, 1).expand(C, H, W)  # shape: C x H x W
#     f_selected_DELTA_l = DELTA_l[mask]

#     # compute boundaries
#     boundaries, a, b = get_boundaries(
#         f_selected_y_star,
#         boundary_base,
#         f_selected_center_v,
#         f_selected_LB,
#         f_selected_UB,
#         f_selected_DELTA_l,
#         T=interval_lower_bound
#     )

#     if y_star is not None:  # encoding
#         bool_indexes = a & b
#         indexes = np.argmax(bool_indexes, axis=0).astype(np.int16)
#         temp_indexes = np.expand_dims(indexes, axis=0)

#         f_selected_LB_new = np.take_along_axis(boundaries, temp_indexes, axis=0)[0]
#         f_selected_UB_new = np.take_along_axis(boundaries, temp_indexes + 1, axis=0)[0]

#         # copy LB, UB for modification
#         LB_new = LB.copy()
#         UB_new = UB.copy()
#         LB_new[mask] = f_selected_LB_new
#         UB_new[mask] = f_selected_UB_new

#     return boundaries, indexes, LB_new, UB_new

def get_boundary_list(mask, y_star, LB, UB, DELTA_l, NUM_TOTAL_SYMBOLS=21, interval_lower_bound=0.3):
    # LB, UB: CHW
    # mask: CHW (bool)
    # DELTA_l: C (1D tensor)

    indexes = None
    LB_new = None
    UB_new = None
    mask = mask.bool()

    # flatten selected elements
    if y_star is not None:
        f_selected_y_star = y_star[mask]  # (N_mask,)
    else:
        f_selected_y_star = None

    f_selected_LB = LB[mask]  # (N_mask,)
    f_selected_UB = UB[mask]  # (N_mask,)

    # compute center and boundary base
    f_selected_center_v, boundary_base = get_boundary_base(
        f_selected_LB, f_selected_UB, NUM_TOTAL_SYMBOLS
    )  # center: (N_mask,), base: (K+1, N_mask)

    # expand DELTA_l: (C,) → (C, H, W)
    C, H, W = LB.shape
    DELTA_l = DELTA_l.view(C, 1, 1).expand(C, H, W)  # (C, H, W)
    f_selected_DELTA_l = DELTA_l[mask]  # (N_mask,)

    # compute boundaries
    boundaries, a, b = get_boundaries(
        f_selected_y_star,
        boundary_base,
        f_selected_center_v,
        f_selected_LB,
        f_selected_UB,
        f_selected_DELTA_l,
        T=interval_lower_bound
    )  # boundaries: (K+1, N_mask)

    if y_star is not None:
        bool_indexes = a & b  # shape: (K, N_mask)
        indexes = torch.argmax(bool_indexes.float(), dim=0).to(torch.int64)  # (N_mask,)
        temp_indexes = indexes.unsqueeze(0)  # shape: (1, N_mask)

        f_selected_LB_new = torch.gather(boundaries, 0, temp_indexes)[0]      # (N_mask,)
        f_selected_UB_new = torch.gather(boundaries, 0, temp_indexes + 1)[0]  # (N_mask,)

        # clone original LB, UB
        LB_new = LB.clone()
        UB_new = UB.clone()
        LB_new[mask] = f_selected_LB_new
        UB_new[mask] = f_selected_UB_new

    return boundaries, indexes, LB_new, UB_new



def get_reconstructions(boundaries, indexes, LB, UB, mask):
    indexes = np.expand_dims(indexes, axis=0)
    f_selected_LB = np.take_along_axis(boundaries, indexes, axis=0)[0]
    f_selected_UB = np.take_along_axis(boundaries, indexes+1, axis=0)[0]

    mask = mask.bool()
    LB[mask] = f_selected_LB  # HWC
    UB[mask] = f_selected_UB  # HWC
    LB_new = LB
    UB_new = UB
    reconstructions = (f_selected_LB + f_selected_UB) / 2
    return reconstructions, LB_new, UB_new

# @jit(nopython=True, parallel=True)
# def erf(x):
#     # save the sign of x
#     sign = np.sign(x)
#     x = np.abs(x)

#     # constants
#     a1 = 0.254829592
#     a2 = -0.284496736
#     a3 = 1.421413741
#     a4 = -1.453152027
#     a5 = 1.061405429
#     p = 0.3275911

#     # A&S formula 7.1.26
#     t = 1.0 / (x * p + 1.0)
#     y = 1.0 - (((((t * a5 + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
#     return sign * y  # erf(-x) = -erf(x)

# @jit(nopython=True, parallel=True)
def cdf(data, sigma):
    return 0.5 * (1 + torch.erf(data / (sigma * 2 ** 0.5)))

# @jit(nopython=True, parallel=True)
def cdf_uniform(boundaries):
    boundaries = boundaries - boundaries[0]
    cdf_list = boundaries / boundaries[-1]
    return cdf_list

# @jit(nopython=True, parallel=True)
# def get_cdf_list(boundaries, pred_sigma):
#     boundaries = boundaries.astype(np.float64)
#     pred_sigma = pred_sigma.astype(np.float64)

#     cdf_list = cdf(boundaries, pred_sigma) - cdf(boundaries[0], pred_sigma)
#     denominator = cdf(boundaries[-1], pred_sigma) - cdf(boundaries[0], pred_sigma)

#     cdf_list *= 1e8
#     denominator *= 1e8

#     denorminator_valid = np.where(denominator > 0, 1, 0)
#     cdf_list = cdf_list / (denominator + (1-denorminator_valid)) # prevent zero division
#     cdf_uniform_list = cdf_uniform(boundaries)
#     cdf_list = cdf_list * denorminator_valid + cdf_uniform_list * (1 - denorminator_valid)
#     cdf_list = np.minimum(cdf_list,1.0)
#     cdf_list = np.maximum(cdf_list, 0.0)
#     return cdf_list

def get_cdf_list(boundaries: torch.Tensor, pred_sigma: torch.Tensor):
    boundaries = boundaries.to(dtype=torch.float64)
    pred_sigma = pred_sigma.to(dtype=torch.float64)

    cdf_list = cdf(boundaries, pred_sigma) - cdf(boundaries[0:1], pred_sigma)  # broadcasting-safe
    denominator = cdf(boundaries[-1:], pred_sigma) - cdf(boundaries[0:1], pred_sigma)

    # scale to integers like original code
    cdf_list = cdf_list * 1e8
    denominator = denominator * 1e8

    # avoid divide-by-zero
    denominator_valid = (denominator > 0).to(boundaries.dtype)
    denominator_safe = denominator + (1.0 - denominator_valid)  # prevents division by zero
    cdf_list = cdf_list / denominator_safe

    # blend with uniform cdf if denominator is invalid
    cdf_uniform_list = cdf_uniform(boundaries)
    cdf_list = cdf_list * denominator_valid + cdf_uniform_list * (1.0 - denominator_valid)

    # clip to [0, 1]
    cdf_list = torch.clamp(cdf_list, min=0.0, max=1.0)

    return cdf_list


def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    # nmaps, _, _ = tensor.shape.as_list()
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + padding // 2, width * xmaps + padding // 2, tensor.shape[3]], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + padding // 2, height - padding
            w, w_width = x * width + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    if ndarr.shape[2] == 1:
        ndarr = ndarr[:,:,0]

    im = Image.fromarray(ndarr)
    im.save(filename)

def save_recon_image(tensor, filename, number_of_blocks_w, number_of_blocks_h,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=number_of_blocks_w, padding=0,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)

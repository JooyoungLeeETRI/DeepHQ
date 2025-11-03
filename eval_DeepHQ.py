import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM_Phase3
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
import numpy as np
from pytorch_msssim import ms_ssim
from PIL import Image
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument(
        "--real", action="store_true", default=False
    )
    parser.add_argument(
        "--num_slices", type=int, default=1,
    )

    parser.add_argument(
        "--continuous_compression", action="store_true", default=False
    )

    parser.add_argument(
        "--for_fid", action="store_true", default=False
    )

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    net = TCM_Phase3(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters (including frozen): {total_params}")

    net = net.to(device)
    net.eval()
    TOTAL_Q = 8
    count = 0
    if args.continuous_compression:
        n_partitions = 23
    else:
        n_partitions = 1
    
    PSNR = np.zeros((TOTAL_Q, n_partitions))
    Bit_rate = np.zeros((TOTAL_Q, n_partitions))
    MS_SSIM = np.zeros((TOTAL_Q, n_partitions))


    total_enc_time = 0
    total_dec_time = 0
    dictory = {}
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
    else:
        print("[Error] Checkpoint option is required.")
        sys.exit(1)
    
    net.update()

    for img_name in img_list:
        img_path = os.path.join(path, img_name)
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
        x = img.unsqueeze(0)
        x_padded, padding = pad(x, p)
        count += 1
        with torch.no_grad():
            if args.cuda:
                torch.cuda.synchronize()
            
            
            #compression
            s = time.time()
            enc_results, NUM_TOTAL_SYMBOLS_K = net.dhq_compress(x_padded, args.continuous_compression)
            e = time.time()
            total_enc_time += (e - s)
            s = time.time()
            dec_results = net.dhq_decompress(enc_results, x_padded.shape, NUM_TOTAL_SYMBOLS_K,args.continuous_compression)
            e = time.time()
            total_dec_time += (e - s)
            if args.cuda:
                torch.cuda.synchronize()
            
            z_string = enc_results["z_string"]
            y_string = enc_results["y_string"]
            total_string_len = len(z_string)

            for q_idx in range(TOTAL_Q):
                y_string_q = y_string[q_idx]
                out_dec_list = dec_results[q_idx]
                for partition_idx, partition_string in enumerate(y_string_q):
                    print(f"len(partition_string): {len(partition_string)}")
                    total_string_len += len(partition_string)
                    out_dec = out_dec_list[partition_idx]
                    out_dec = crop(out_dec, padding)
                    num_pixels = x.size(0) * x.size(2) * x.size(3)

                    # Compute bpp
                    bpp_val = total_string_len * 8.0 / num_pixels
                    Bit_rate[q_idx][partition_idx] += bpp_val

                    # === Save reconstructed image ===
                    if args.for_fid:
                        # Save inside rate-specific directory, file name same as original
                        save_dir = os.path.join("recon_results_for_fid", os.path.basename(args.data.rstrip("/")),
                                                f"q{q_idx:02d}_p{partition_idx:02d}")
                        os.makedirs(save_dir, exist_ok=True)

                        save_path = os.path.join(save_dir, img_name)  # Use original image name
                        out_img = transforms.ToPILImage()(out_dec.squeeze().cpu().clamp(0, 1))
                        out_img.save(save_path)
                        print(f"Saved for measuring FID: {save_path}")
                    else:
                        psnr_val = compute_psnr(x, out_dec)
                        msssim_val = compute_msssim(x, out_dec)

                        PSNR[q_idx][partition_idx] += psnr_val
                        MS_SSIM[q_idx][partition_idx] += msssim_val
                        
                        print(f'Quality level:{q_idx}, partiontion idx:{partition_idx}')
                        print(f'Bitrate: {(total_string_len * 8.0 / num_pixels):.3f}bpp')
                        print(f'MS-SSIM: {msssim_val:.2f}dB')
                        print(f'PSNR: {psnr_val:.2f}dB')
                       
                        save_dir = os.path.join("recon_results", os.path.basename(args.data.rstrip("/")), img_name)
                        os.makedirs(save_dir, exist_ok=True)

                        # Add postfix [BPP][PSNR] to filename
                        save_path = os.path.join(save_dir, f"q{q_idx:02d}_p{partition_idx:02d}_[{bpp_val:.4f}bpp][{psnr_val:.3f}dB].png")

                        out_img = transforms.ToPILImage()(out_dec.squeeze().cpu().clamp(0, 1))
                        out_img.save(save_path)
                        print(f"Saved reconstructed image: {save_path}")


    avg_enc_time = total_enc_time / (count * ((TOTAL_Q-1) * n_partitions + 1))
    avg_dec_time = total_dec_time / (count * ((TOTAL_Q-1) * n_partitions + 1))

    if args.for_fid:
        Bit_rate = Bit_rate / count
        avg_enc_time = total_enc_time / (count * ((TOTAL_Q-1) * n_partitions + 1))
        avg_dec_time = total_dec_time / (count * ((TOTAL_Q-1) * n_partitions + 1))
        # total number of decoded images: count * ((TOTAL_Q-1) * n_partitions + 1)
        for q_idx in range(TOTAL_Q):
            print(f'Quality level:{q_idx} ############')
            if q_idx == 0:
                N = 1
            else:
                N = n_partitions

            for partition_idx in range(N):
                print(f'average_Bit-rate: {Bit_rate[q_idx][partition_idx]:.3f} bpp')

    else:
        PSNR = PSNR / count
        MS_SSIM = MS_SSIM / count
        Bit_rate = Bit_rate / count
        for q_idx in range(TOTAL_Q):
            print(f'Quality level:{q_idx} ############')
            if q_idx == 0:
                N = 1
            else:
                N = n_partitions

            for partition_idx in range(N):
                print(f'partition idx:{partition_idx}')
                print(f'average_PSNR: {PSNR[q_idx][partition_idx]:.2f}dB')
                print(f'average_MS-SSIM: {MS_SSIM[q_idx][partition_idx]:.4f}')
                print(f'average_Bit-rate: {Bit_rate[q_idx][partition_idx]:.3f} bpp')
    
        print("########################")
        print(f'average_enc_time: {avg_enc_time:.3f} s')
        print(f'average_dec_time: {avg_dec_time:.3f} s')
        print(f'total number of decoded images: {count * ((TOTAL_Q-1) * n_partitions + 1)}')
        print("########################")
    
    csv_filename = f"{os.path.basename(args.data.rstrip('/'))}_quality_metrics.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Quality Level", "Bit-rate (bpp)", "PSNR (dB)", "MS_SSIM"])
        for q_idx in range(TOTAL_Q):
            if q_idx == 0:
                N = 1
            else:
                N = n_partitions
            for partition_idx in range(N):
                writer.writerow([q_idx, f"{Bit_rate[q_idx][partition_idx]:.6f}", f"{PSNR[q_idx][partition_idx]:.4f}",  f"{MS_SSIM[q_idx][partition_idx]:.6f}"])

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
    
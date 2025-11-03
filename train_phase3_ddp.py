import os
import sys
import random
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from compressai.zoo import models
from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim
from models import TCM_Phase3
from torch.utils.tensorboard import SummaryWriter


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)


class MultiRateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=[0.0025,0.0035,0.0067,0.013,0.025,0.05,0.1,0.2], type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

        # {
        #     "x_hat_list": x_hat_list,
        #     "likelihoods": {"y": y_likelihoods_list, "z": z_likelihoods},
        #     # "para":{"means": means, "scales":scales, "y":y}
        # }
    def forward(self, output_list, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out_list = []
        for q_idx, output in enumerate(output_list):
            out = {}    
            #APPLY MASK!
            # out["bpp_loss"] = sum(
            #     (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            #     for likelihoods in output["likelihoods"].values()
            # )
            log2 = math.log(2)
            log_likelihood_y = torch.log(output["likelihoods"]["y"])
            masked_log_likelihood_y = log_likelihood_y * output["mask"]
            bpp_y = masked_log_likelihood_y.sum() / (-log2 * num_pixels)
            log_likelihood_z = torch.log(output["likelihoods"]["z"])
            bpp_z = log_likelihood_z.sum() / (-log2 * num_pixels)
            out["bpp_loss"] = bpp_y + bpp_z
            out["bpp_loss_y"] = bpp_y
            out["bpp_loss_z"] = bpp_z

            if self.type == 'mse':
                out["mse_loss"] = self.mse(output["x_hat"], target)
                out["loss"] = self.lmbda[q_idx] * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
            else:
                out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
                out["loss"] = self.lmbda[q_idx] * (1 - out['ms_ssim_loss']) + out["bpp_loss"]
            out_list.append(out)
        out_mean = {}
        loss_mean = sum(out["loss"] for out in out_list) / len(out_list)
        bpp_mean = sum(out["bpp_loss"] for out in out_list) / len(out_list)
        out_mean["loss_mean"] = loss_mean
        out_mean["bpp_mean"] = bpp_mean
        if self.type == 'mse':
            mse_mean = sum(out["mse_loss"] for out in out_list) / len(out_list)
            out_mean["mse_mean"] = mse_mean
        else:
            ms_ssim_mean = sum(out["ms_ssim_loss"] for out in out_list) / len(out_list)
            out_mean["ms_ssim_mean"] = ms_ssim_mean

        return out_mean, out_list


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def configure_optimizers(net, args):
#     """Separate parameters for the main optimizer and the auxiliary optimizer.
#     Return two optimizers"""

#     parameters = {
#         n
#         for n, p in net.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad
#     }
#     aux_parameters = {
#         n
#         for n, p in net.named_parameters()
#         if n.endswith(".quantiles") and p.requires_grad
#     }

#     # Make sure we don't have an intersection of parameters
#     params_dict = dict(net.named_parameters())
#     inter_params = parameters & aux_parameters
#     union_params = parameters | aux_parameters

#     assert len(inter_params) == 0
#     assert len(union_params) - len(params_dict.keys()) == 0

#     optimizer = optim.Adam(
#         (params_dict[n] for n in sorted(parameters)),
#         lr=args.learning_rate,
#     )
#     aux_optimizer = optim.Adam(
#         (params_dict[n] for n in sorted(aux_parameters)),
#         lr=args.aux_learning_rate,
#     )
#     return optimizer, aux_optimizer

def configure_optimizers(net, args):
    params_dict = dict(net.named_parameters())

    # Group 1: regular parameters (excluding quantiles and net.gamma)
    base_params = []
    # Group 2: net.gamma only (higher learning rate)
    gamma_params = []
    # Group 3: quantiles (for auxiliary optimizer)
    aux_params = []

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".quantiles"):
            aux_params.append(param)
        elif name == "gamma" or name == "module.gamma":
            gamma_params.append(param)
        else:
            base_params.append(param)

    print("gamma_params:")
    print(gamma_params)
    # Main optimizer with two parameter groups
    optimizer = optim.Adam([
        {"params": base_params, "lr": args.learning_rate},
        {"params": gamma_params, "lr": args.learning_rate * 1000}
    ])

    # Auxiliary optimizer for quantiles (if any)
    aux_optimizer = None
    if len(aux_params) > 0:
        aux_optimizer = optim.Adam(aux_params, lr=args.aux_learning_rate)

    return optimizer, aux_optimizer



def setup_ddp():
    print(f"[setup_ddp] init_process_group called")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[setup_ddp] done setting device to cuda:{local_rank}")


def cleanup_ddp():
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=20,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-5,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=5e-5,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=False
    )
    # args = parser.parse_args(argv)
    args = parser.parse_args(sys.argv[1:])
    return args


def test_epoch(epoch, test_dataloader, model, criterion, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_mean, out_list = criterion(out_net, d)                
                aux_loss.update(model.module.aux_loss())
                bpp_loss.update(out_mean["bpp_mean"])
                loss.update(out_mean["loss_mean"])
                mse_loss.update(out_mean["mse_mean"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMean MSE loss: {mse_loss.avg:.3f} |"
            f"\tMean Bpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_mean, out_list = criterion(out_net, d)                
                aux_loss.update(model.module.aux_loss())
                bpp_loss.update(out_mean["bpp_mean"])
                loss.update(out_mean["loss_mean"])
                ms_ssim_loss.update(out_mean["ms_ssim_mean"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMean MS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tMean Bpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    return loss.avg


def main_worker(rank, world_size, args):
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    type = args.type

    save_path = args.save_path

    model = TCM_Phase3(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)

    # for param in model.parameters():
    #     param.requires_grad = False
    # # Unfreeze h_importance_s
    # for name, param in model.h_importance_s.named_parameters():
    #     param.requires_grad = True
    #     print(f"[TRAINABLE] h_importance_s.{name}")
    # # Unfreeze lrp_transforms
    # for name, param in model.lrp_transforms.named_parameters():
    #     param.requires_grad = True
    #     print(f"[TRAINABLE] lrp_transforms.{name}")
    # model.gamma.requires_grad = True
    

    ###############################################
    ################# for debug ###################
    def add_nan_hook(m, name):
        def fwd_hook(module, args, output):
            # If output is a tuple, check each element
            if isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        if torch.isnan(o).any() or torch.isinf(o).any():
                            print(f"[NaN/Inf] in {module.__class__.__name__} output[{i}]")
            elif isinstance(output, torch.Tensor):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"[NaN/Inf] in {module.__class__.__name__} output")

        m.register_forward_hook(fwd_hook)

    for name, module in model.named_modules():
        add_nan_hook(module, name)

    def grad_nan_hook(name):
        def hook(grad):
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"[NaN/Inf GRAD] {name}")
        return hook

    for n, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(grad_nan_hook(n))
    #########################################################
    #########################################################


    model = model.to(device)
    print(f"[RANK {rank}] reached point A on device {device}")
    
    torch.distributed.barrier()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    torch.distributed.barrier()
    print(f"[RANK {rank}] reached point B")


    last_epoch = 0
    optimizer, aux_optimizer = configure_optimizers(model, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.2, last_epoch=-1)

    criterion = MultiRateDistortionLoss(type=args.type)

    if hasattr(args, 'checkpoint') and args.checkpoint:
        if rank == 0:
            print("Loading", args.checkpoint)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.checkpoint, map_location=map_location)

        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model_state_dict = model.state_dict()
        
        def adapt_state_dict_for_model(state_dict, model_state_dict):
            def has_module_prefix(state_dict):
                return all(k.startswith("module.") for k in state_dict.keys())
            ckpt_has_module = has_module_prefix(state_dict)
            model_has_module = has_module_prefix(model_state_dict)
            if ckpt_has_module and not model_has_module:
                # Case 2: remove "module." from checkpoint keys
                adapted = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            elif not ckpt_has_module and model_has_module:
                # Case 3: add "module." to checkpoint keys
                adapted = {f"module.{k}": v for k, v in state_dict.items()}
            else:
                # Case 1 or 4: no change needed
                adapted = state_dict
            return adapted
        
        state_dict = adapt_state_dict_for_model(state_dict, model_state_dict)

        # for k in state_dict.keys():
        #     print(k)
        # asdfasdf

        # Keep only parameters with matching keys and shapes
        matched_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state_dict
        }

        # Load the matched parameters (ignore missing/unexpected keys)
        model_core = getattr(model, "module", model)
        load_result = model_core.load_state_dict(matched_state_dict, strict=False)
        # load_result = net.load_state_dict(matched_state_dict, strict=False)

        print(f"Loaded {len(matched_state_dict)} parameters from checkpoint.")
        if load_result.missing_keys and rank == 0:
            print("Missing keys:", load_result.missing_keys)
        if load_result.unexpected_keys:
            print("Unexpected keys:", load_result.unexpected_keys)
        
        if hasattr(args, 'continue_train') and args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            if aux_optimizer is not None:
                aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    else:
        print("Warning: Checkpoint does not exist. Exiting...")
        sys.exit(1)

        if hasattr(args, 'continue_train') and args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])  # if used


    train_transform = transforms.Compose([
        transforms.RandomCrop(args.patch_size), transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(args.patch_size), transforms.ToTensor()
    ])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transform)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    writer = None
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.save_path, "tensorboard"))

    best_loss = float("inf")

    for epoch in range(last_epoch, args.epochs):
        if epoch == args.epochs - 4:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2e-6
            print(f"[Epoch {epoch}] LR manually set to 2e-6")
        elif epoch == args.epochs - 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 4e-7
            print(f"[Epoch {epoch}] LR manually set to 4e-7")
        print(f"[Epoch {epoch}] Learning rate: {optimizer.param_groups[0]['lr']}")
        train_sampler.set_epoch(epoch)
        model.train()
        device = rank

        for i, d in enumerate(train_loader):
            d = d.to(device)
            optimizer.zero_grad()
            if aux_optimizer is not None:
                aux_optimizer.zero_grad()

            outnet_list = model(d)
            out_mean, out_list = criterion(outnet_list, d)
            
            assert not torch.isnan(out_mean["loss_mean"]), "loss is NaN"
            out_mean["loss_mean"].backward()
            # with torch.autograd.set_detect_anomaly(True):
            #     out_mean["loss_mean"].backward()

            # # Check for NaN or Inf
            # has_nan_or_inf = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
            # if has_nan_or_inf:
            #     print("NaN or Inf in gradients.")
            #     print(f"out_mean[loss_mean]:{out_mean['loss_mean']}")
            #     for i, out in enumerate(out_list):
            #         print(f"out[{i}][loss]:{out['loss']}")
            #         print(f"out[{i}][mse]:{out['mse_loss']}") 
            #         print(f"out[{i}][bpp]:{out['bpp_loss']}") 
            #         print(f"out[{i}][bpp_y]:{out['bpp_loss_y']}") 
            #         print(f"out[{i}][bpp_z]:{out['bpp_loss_z']}") 
            #     sys.exit(1)
            # else:
            #     if args.clip_max_norm > 0:
            #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            #     optimizer.step()
            
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            if aux_optimizer is not None:
                aux_loss = model.module.aux_loss()
                aux_loss.backward()
                aux_optimizer.step()
            else:
                aux_loss = torch.tensor(0)

            if rank == 0 and i % 100 == 0:
                world_size = dist.get_world_size()
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)*world_size}/{len(train_loader.dataset)}"
                    f" ({100. * i / len(train_loader):.0f}%)]"
                    f'\tLoss: {out_mean["loss_mean"]:.3f} |'
                )
                for q_idx, out_criterion in enumerate(out_list):
                    if type == 'mse':
                        print(
                            f'Q: {q_idx} |'
                            f'\tMask True Ratio: {outnet_list[q_idx]["mask"].float().mean().item():.3f} |'
                            f'\tgamma[{q_idx}] mean: {model.module.gamma[q_idx].mean().item()} |'
                            f'\tLoss: {out_criterion["loss"].item():.3f} |'
                            f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                            f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                            f"\tAux loss: {aux_loss.item():.2f}"
                        )
                        
                    else:
                        print(
                            f'Q: {q_idx} |'
                            f'\tMask True Ratio: {outnet_list[q_idx]["mask"].float().mean().item():.3f} |'
                            f'\tgamma[{q_idx}] mean: {model.gamma[q_idx].mean().item()} |'
                            f'\tLoss: {out_criterion["loss"].item():.3f} |'
                            f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                            f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                            f"\tAux loss: {aux_loss.item():.2f}"
                        )

        # model.eval()
        # loss_meter = AverageMeter()
        # with torch.no_grad():
        #     for d in test_loader:
        #         d = d.to(device)
        #         out_net = model(d)
        #         out_mean, out_list = criterion(out_net, d)
        #         loss_meter.update(out_mean["loss_mean"].item(), d.size(0))

        lr_scheduler.step()
        if rank == 0:
            # print(f"Test epoch {epoch}: Average Loss: {loss_meter.avg:.4f}")
            # writer.add_scalar('Loss/test', loss_meter.avg, epoch)
            loss = test_epoch(epoch, test_loader, model, criterion, type)
            writer.add_scalar('test_loss', loss, epoch)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            os.makedirs(args.save_path, exist_ok=True)
            if aux_optimizer is not None:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    epoch,
                    save_path,
                    save_path + str(epoch) + "_checkpoint.pth.tar",
                )
            else:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    epoch,
                    save_path,
                    save_path + str(epoch) + "_checkpoint.pth.tar",
                )
            
            # ckpt = {
            #     "epoch": epoch,
            #     "state_dict": model.module.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            #     "aux_optimizer": aux_optimizer.state_dict(),
            # }
            # torch.save(ckpt, os.path.join(args.save_path, f"epoch_{epoch}.pth"))
            # if is_best:
            #     torch.save(ckpt, os.path.join(args.save_path, "checkpoint_best.pth"))

    cleanup_ddp()

def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")


def main():
    args = parse_args()
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    main_worker(local_rank, world_size, args)


if __name__ == "__main__":
    main()

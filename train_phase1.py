import argparse
import math
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import TCM
from torch.utils.tensorboard import SummaryWriter   
import os

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, type='mse'
):
    model.train()
    device = next(model.parameters()).device

    # for i, d in enumerate(train_dataloader):
    #     print(f"[{epoch}][{i}] batch loaded")
    #     d = d.to(device)
    #     print(f"[{epoch}][{i}] moved to device")

    #     optimizer.zero_grad()
    #     aux_optimizer.zero_grad()

    #     print(f"[{epoch}][{i}] forward start")
    #     out_net = model(d)
    #     print(f"[{epoch}][{i}] forward done")

    #     out_criterion = criterion(out_net, d)
    #     print(f"[{epoch}][{i}] loss computed")

    #     out_criterion["loss"].backward()
    #     print(f"[{epoch}][{i}] main backward done")

    #     if clip_max_norm > 0:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
    #     optimizer.step()
    #     print(f"[{epoch}][{i}] optimizer step done")

    #     aux_loss = model.aux_loss()
    #     print(f"[{epoch}][{i}] aux loss computed")

    #     aux_loss.backward()
    #     print(f"[{epoch}][{i}] aux backward done")

    #     aux_optimizer.step()
    #     print(f"[{epoch}][{i}] aux optimizer step done")
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(i)
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )


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
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
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
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    return loss.avg


def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")


def parse_args(argv):
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
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
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
        default=1e-3,
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
    args = parser.parse_args(argv)
    return args

def main(argv):    
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "tensorboard/")

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )


    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)
    net = net.to(device)

    # dummy = torch.randn(1, 3, *args.patch_size).to(device)
    # with torch.no_grad():
    #     net(dummy)
    # net.update(force=True)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.2, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0
    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)

    # Extract raw state_dict from checkpoint
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_state_dict = net.state_dict()

    # Detect prefix pattern
    ckpt_keys_have_module = all(k.startswith("module.") for k in state_dict.keys())
    model_keys_have_module = all(k.startswith("module.") for k in model_state_dict.keys())

    def adjust_key_prefix(k):
        if ckpt_keys_have_module and not model_keys_have_module:
            return k.replace("module.", "", 1)
        elif not ckpt_keys_have_module and model_keys_have_module:
            return "module." + k
        return k
    
    # Adjust checkpoint keys to match current model's state_dict
    adjusted_state_dict = {adjust_key_prefix(k): v for k, v in state_dict.items()}
    

    if args.continue_train:
        # Load full model (resume)
        model_core = getattr(net, "module", net)
        load_result = model_core.load_state_dict(adjusted_state_dict, strict=True)
        # load_result = net.load_state_dict(adjusted_state_dict, strict=True)
        print("✅ Loaded full model state_dict for continued training.")
    else:
        # Load only selected modules from pretrained checkpoint
        target_prefixes = [
            'g_a', 'g_s', 'h_a', 'h_mean_s', 'h_scale_s',
            'entropy_bottleneck', 'gaussian_conditional'
        ]
        filtered_state_dict = {
            k: v for k, v in adjusted_state_dict.items()
            if any(k.startswith(p) for p in target_prefixes)
        }
        matched_state_dict = {
            k: v for k, v in filtered_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_core = getattr(net, "module", net)
        load_result = model_core.load_state_dict(matched_state_dict, strict=False)
        # load_result = net.load_state_dict(matched_state_dict, strict=False)
        print(f"✅ Loaded {len(matched_state_dict)} selected parameters from pretrained checkpoint.")

    # Report missing/unexpected keys
    if load_result is not None:
        if load_result.missing_keys:
            print("⚠️ Missing keys:", load_result.missing_keys)
        if load_result.unexpected_keys:
            print("⚠️ Unexpected keys:", load_result.unexpected_keys)

    # If continuing training, restore optimizer and scheduler
    if args.continue_train:
        last_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


    # if args.checkpoint:
    #     print("Loading", args.checkpoint)
    #     checkpoint = torch.load(args.checkpoint, map_location=device)

    # # Determine whether to load full model or selective weights
    # if args.continue_train:
    #     # Resuming training of current model — load full checkpoint
    #     load_result = net.load_state_dict(checkpoint["state_dict"], strict=True)
    #     print("Loaded full model state_dict for continued training.")
    # else:
    #     # Initializing model from pretrained TCM model — load selected modules only
    #     state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    #     target_prefixes = [
    #         'g_a',
    #         'g_s',
    #         'h_a',
    #         'h_mean_s',
    #         'h_scale_s',
    #         'entropy_bottleneck',
    #         'gaussian_conditional',
    #     ]
    #     filtered_state_dict = {
    #         k: v for k, v in state_dict.items()
    #         if any(k.startswith(p) for p in target_prefixes)
    #     }
    #     net_state_dict = net.state_dict()
    #     matched_state_dict = {
    #         k: v for k, v in filtered_state_dict.items()
    #         if k in net_state_dict and net_state_dict[k].shape == v.shape
    #     }
    #     load_result = net.load_state_dict(matched_state_dict, strict=False)
    #     print(f"Loaded {len(matched_state_dict)} selected parameters from A-model checkpoint.")

    # # Report missing/unexpected keys
    # if load_result is not None:
    #     if load_result.missing_keys:
    #         print("Missing keys:", load_result.missing_keys)
    #     if load_result.unexpected_keys:
    #         print("Unexpected keys:", load_result.unexpected_keys)

    # # If continuing training, restore optimizer and scheduler states
    # if args.continue_train:
    #     last_epoch = checkpoint["epoch"] + 1
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
    #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            type
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, type)
        writer.add_scalar('test_loss', loss, epoch)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            model_core = getattr(net, "module", net)
            model_core.update(force=True)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
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


if __name__ == "__main__":
    main(sys.argv[1:])
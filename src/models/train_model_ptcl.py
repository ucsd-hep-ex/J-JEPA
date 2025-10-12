import os
import sys
import logging
import argparse
from pathlib import Path
import json
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data._utils.collate import default_collate
import torch.distributed as dist
import time
import random

import torch.cuda as cuda

from src.options import Options
from src.models.jjepa import JJEPA
from src.dataset.ParticleDataset import ParticleDataset
from src.util.create_random_masks import create_random_masks
from src.util.cov_loss import covariance_loss
from src.util.var_loss import variance_loss

import math
from torch.nn.parallel import DistributedDataParallel as DDP

def unwrap(m):
    return m.module if isinstance(m, DDP) else m

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train JJEPA model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="/mnt/d/physic/I-JEPA-Jets-Subash/src/test_options.json",
        help="Path to config JSON file",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Start training from a saved checkpoint",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of gpus")
    parser.add_argument(
        "--num_jets", type=int, default=1200 * 1000, help="Number of jets to train on"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument(
        "--cov_loss_weight", type=float, default=0.0, help="covariance loss weight"
    )
    parser.add_argument(
        "--var_loss_weight", type=float, default=0.0, help="variance loss weight"
    )
    parser.add_argument(
        "--var_flatten",
        type=int,
        default=1,
        help="flatten reps when calculating variance loss",
    )
    parser.add_argument(
        "--base_momentum",
        type=float,
        default=0.99,
        help="base momentum for momentum scheduler",
    )
    return parser.parse_args()


def setup_environment(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")


def setup_data_loader(args, options, data_path, world_size, rank, tag="train"):
    if tag == "val":
        data_path = data_path.replace("train", "val")
        dataset = ParticleDataset(data_path, num_jets=options.num_val_jets)
    else:
        dataset = ParticleDataset(data_path, num_jets=options.num_jets)

    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=(tag == "train"), drop_last=(tag == "train")
        )

    stats = dataset.stats
    shuffle = (sampler is None) and (tag == "train")

    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=shuffle,
        num_workers=options.num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_fn,
        persistent_workers=True if options.num_workers > 0 else False,
    )
    return loader, sampler, len(dataset), stats

def collate_fn(batch):
    tensors = default_collate([b[:-1] for b in batch])
    subjets = [b[-1] for b in batch]
    return (*tensors, subjets)

def ddp_setup_if_needed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        return True, local_rank, world_size
    return False, 0, 1


def save_checkpoint(model, optimizer, epoch, loss_train, loss_val, output_dir):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "training loss": loss_train,
        "validation loss": loss_val,
    }
    torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth"))


logger = logging.getLogger(__name__)

def setup_logging(rank, output_dir):
    log_file = Path(output_dir) / f"train_rank_{rank}.log"

    global logger
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # console only on rank 0
    if rank == 0:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)


class AverageMeter(object):
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


def create_momentum_scheduler(options):
    return cosine_scheduler(
        options.base_momentum, 1, options.num_epochs, options.num_steps_per_epoch
    )


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return iter(schedule)


def gpu_timer(closure):
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.time()
    result = closure()
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        elapsed_time = (time.time() - start) * 1000
    return result, elapsed_time


def log_gpu_stats(device):
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved  = torch.cuda.memory_reserved(device)  / 1024**3
        logger.info(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
        logger.info(f"GPU Memory Reserved:  {memory_reserved:.2f} GB")


def main(rank, world_size, args):
    torch.cuda.set_device(rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")

    out_dir = args.output_dir
    if os.path.isdir(out_dir):
        contents = os.listdir(out_dir)
        non_log_files = [file for file in contents if file.endswith(".pth")]
        if non_log_files:
            sys.exit(
                "ERROR: experiment already exists and contains files other than log files; don't want to overwrite it by mistake"
            )
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(f"cuda:{rank}")
    args.num_val_jets = args.num_jets // 4

    options = Options.load(args.config)
    options.batch_size = args.batch_size
    options.num_steps_per_epoch = math.ceil(args.num_jets / (args.batch_size * world_size)) # match DDP for EMA updates
    options.cov_loss_weight = args.cov_loss_weight
    options.var_loss_weight = args.var_loss_weight
    options.base_momentum = args.base_momentum
    options.encoder_pos_emb = False

    setup_logging(rank, args.output_dir)
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    if options.use_parT_encoder:
        logger.info("Using ParT Encoder")
    else:
        logger.info("Using JetTransformer Encoder")
    if options.use_parT_predictor:
        logger.info("Using ParT Predictor")
    else:
        logger.info("Using JetTransformer Predictor")
    logger.info(f"covariance loss weight: {options.cov_loss_weight}")
    logger.info(f"variance loss weight:  {options.var_loss_weight}")
    logger.info(f"base momentum:         {options.base_momentum}")

    model = JJEPA(options).to(device)
    model = model.to(dtype=torch.float32)

    def check_for_nan(module, input, output):
        for idx, inp in enumerate(input):
            if torch.is_tensor(inp) and (torch.isnan(inp).any() or torch.isinf(inp).any()):
                print(f"NaN or Inf detected in input {idx} of {module}")
        if torch.is_tensor(output):
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"NaN or Inf detected in output of {module}")
        elif isinstance(output, tuple):
            for idx, out in enumerate(output):
                if torch.is_tensor(out) and (torch.isnan(out).any() or torch.isinf(out).any()):
                    print(f"NaN or Inf detected in output {idx} of {module}")
        else:
            print(f"Output of {module} is neither Tensor nor Tuple")

    for _, module in model.named_modules():
        module.register_forward_hook(check_for_nan)

    logger.info(model)

    checkpoint = {}
    if args.load_checkpoint and Path(args.load_checkpoint).is_file():
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        logger.info(f"Loaded model from {args.load_checkpoint}")

    param_groups = [
        {
            "params": (
                p
                for n, p in model.context_transformer.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in model.predictor_transformer.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in model.context_transformer.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in model.predictor_transformer.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]
    for p in model.target_transformer.parameters():
        p.requires_grad = False

    if args.lr:
        options.lr = args.lr

    optimizer = optim.AdamW(
        param_groups,
        lr=options.lr,
        weight_decay=options.weight_decay,
        eps=options.eps,
    )
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"Loaded optimizer state from {args.load_checkpoint}")

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options.num_epochs)
    scaler = GradScaler()
    momentum_scheduler = create_momentum_scheduler(options)

    train_loader, train_sampler, train_dataset_size, train_stats = setup_data_loader(
        args, options, args.data_path, world_size, rank, tag="train"
    )
    val_loader, val_sampler, val_dataset_size, val_stats = setup_data_loader(
        args, options, args.data_path, world_size, rank, tag="val"
    )
    logger.info(f"Train dataset size: {train_dataset_size}")
    logger.info(f"Val dataset size:   {val_dataset_size}")

    losses_train, mse_losses_train, var_losses_train, cov_losses_train = [], [], [], []
    losses_val,   mse_losses_val,   var_losses_val,   cov_losses_val   = [], [], [], []
    lowest_val_loss = np.inf

    for epoch in range(options.start_epochs, options.num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        logger.info("lr: %f" % scheduler.get_last_lr()[0])
        epoch_start_time = time.time()

        if train_sampler:
            train_sampler.set_epoch(epoch)
        if val_sampler:
            val_sampler.set_epoch(epoch)

        loss_meter_train = AverageMeter()
        mse_loss_meter_train = AverageMeter()
        cov_loss_meter_train = AverageMeter()
        var_loss_meter_train = AverageMeter()
        mse_loss_meter_val = AverageMeter()
        cov_loss_meter_val = AverageMeter()
        var_loss_meter_val = AverageMeter()
        loss_meter_val = AverageMeter()
        time_meter_train = AverageMeter()
        time_meter_val = AverageMeter()

        steps_train = math.ceil(train_dataset_size / (options.batch_size * world_size))
        pbar_t = tqdm(
            train_loader,
            total=steps_train,
            desc="Training",
            disable=(rank != 0),
        )

        model.train()
        for itr, (p4_spatial, p4, particle_mask, subjets) in enumerate(pbar_t):
            start_data_loading = time.time()
            particle_mask = particle_mask.squeeze(-1).bool()
            p4 = p4.to(dtype=torch.float32)
            p4_spatial = p4_spatial.to(dtype=torch.float32)
            p4 = p4.to(device, non_blocking=True)
            p4_spatial = p4_spatial.to(device, non_blocking=True)
            particle_mask = particle_mask.to(device, non_blocking=True, dtype=torch.float32)

            particle_mask_cpu = particle_mask.cpu().bool()

            total_real_particles = particle_mask_cpu.sum(dim=1)
            multi_real = total_real_particles > 1

            if not multi_real.all():
                valid = multi_real.nonzero(as_tuple=True)[0]
                p4_spatial = p4_spatial[valid]
                p4 = p4[valid]
                particle_mask = particle_mask[valid]
                particle_mask_cpu = particle_mask_cpu[valid]

            while True:
                context_masks, target_masks = create_random_masks(
                    p4_spatial,
                    subjets,
                    ratio=options.trgt_ratio,
                    max_targets=options.max_targets,
                )
                real_context_counts = (context_masks & particle_mask_cpu).sum(dim=1)
                real_target_counts = (target_masks & particle_mask_cpu).sum(dim=1)
                if (real_context_counts > 0).all() and (real_target_counts > 0).all():
                    break

            context_masks = context_masks.to(device)
            target_masks = target_masks.to(device)

            context_masks_expanded = context_masks.unsqueeze(-1).expand(-1, -1, 4)
            target_masks_expanded  = target_masks.unsqueeze(-1).expand(-1, -1, 4)

            end_data_loading = time.time()
            logger.info(f"Data loading time for batch {itr}: {end_data_loading - start_data_loading:.3f} seconds")

            start_forward_pass = time.time()

            def train_step():
                cov_l = 0
                var_l = 0
                optimizer.zero_grad()
                with autocast(enabled=options.use_amp):
                    B = p4_spatial.shape[0]
                    N_ctxt = context_masks.sum(dim=1).max().item()
                    N_trgt = target_masks.sum(dim=1).max().item()
                    p4_context = p4[context_masks_expanded].view(B, N_ctxt, 4)
                    p4_target  = p4[target_masks_expanded].view(B, N_trgt, 4)
                    ctxt_particle_mask = particle_mask[context_masks].view(B, N_ctxt)
                    trgt_particle_mask = particle_mask[target_masks].view(B, N_trgt)

                    context = {"p4": p4_context, "particle_mask": ctxt_particle_mask, "split_mask": context_masks}
                    target  = {"p4": p4_target,  "particle_mask": trgt_particle_mask, "split_mask": target_masks}
                    full_jet = {"p4": p4, "p4_spatial": p4_spatial, "particle_mask": particle_mask}

                    pred_repr, target_repr, context_repr = model(context, target, full_jet, train_stats)
                    mse_loss = nn.functional.mse_loss(pred_repr, target_repr)
                    loss = mse_loss.clone()

                    if options.cov_loss_weight > 0 or options.var_loss_weight > 0:
                        context_mask_expanded = ctxt_particle_mask.unsqueeze(-1)
                        masked_context_reps = context_repr * context_mask_expanded
                        target_mask_expanded = trgt_particle_mask.unsqueeze(-1)
                        masked_target_reps = target_repr * target_mask_expanded
                        if options.cov_loss_weight > 0:
                            cov_l = (covariance_loss(target_repr) + covariance_loss(context_repr)) / 2
                            loss += options.cov_loss_weight * cov_l
                        if options.var_loss_weight > 0:
                            var_l = (variance_loss(masked_target_reps, trgt_particle_mask) +
                                     variance_loss(masked_context_reps, ctxt_particle_mask)) / 2
                            loss += options.var_loss_weight * var_l

                    if options.use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        if options.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), options.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if options.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), options.max_grad_norm)
                        optimizer.step()

                    with torch.no_grad():
                        m = next(momentum_scheduler)
                        for param_q, param_k in zip(
                            unwrap(model).context_transformer.parameters(),
                            unwrap(model).target_transformer.parameters(),
                        ):
                            param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                loss_dict = {
                    "total_loss": float(loss),
                    "mse_loss": float(mse_loss),
                    "cov_loss": float(cov_l) if options.cov_loss_weight > 0 else 0,
                    "var_loss": float(var_l) if options.var_loss_weight > 0 else 0,
                }
                return loss_dict

            loss_dict, etime = gpu_timer(train_step)

            end_forward_pass = time.time()
            logger.info(f"Forward pass time for batch {itr}: {end_forward_pass - start_forward_pass:.3f} seconds")

            loss_meter_train.update(loss_dict["total_loss"])
            mse_loss_meter_train.update(loss_dict["mse_loss"])
            if options.cov_loss_weight > 0:
                cov_loss_meter_train.update(loss_dict["cov_loss"])
            if options.var_loss_weight > 0:
                var_loss_meter_train.update(loss_dict["var_loss"])
            time_meter_train.update(etime)

            if rank == 0 and itr % options.log_freq == 0:
                logger.info(f"[{epoch + 1}, {itr}] total training loss: {loss_meter_train.avg:.3f}, ({time_meter_train.avg:.1f} ms)")
                logger.info(f"mse loss: {mse_loss_meter_train.avg:+.3f}, cov loss: {cov_loss_meter_train.avg:+.3f}, var loss: {var_loss_meter_train.avg:+.3f}")
                log_gpu_stats(device)

        train_time_end = time.time()
        

        steps_val = math.ceil(val_dataset_size / (options.batch_size * world_size))
        pbar_v = tqdm(
            val_loader,
            total=steps_val,
            desc="Validation",
            disable=(rank != 0),
        )

        for itr, (p4_spatial, p4, particle_mask, subjets) in enumerate(pbar_v):
            particle_mask = particle_mask.squeeze(-1).bool()
            p4 = p4.to(dtype=torch.float32)
            p4_spatial = p4_spatial.to(dtype=torch.float32)
            p4 = p4.to(device, non_blocking=True)
            p4_spatial = p4_spatial.to(device, non_blocking=True)
            particle_mask = particle_mask.to(device, non_blocking=True, dtype=torch.float32)

            particle_mask_cpu = particle_mask.cpu().bool()
            total_real_particles = particle_mask_cpu.sum(dim=1)
            multi_real = total_real_particles > 1
            if not multi_real.all():
                valid = multi_real.nonzero(as_tuple=True)[0]
                p4_spatial = p4_spatial[valid]
                p4 = p4[valid]
                particle_mask = particle_mask[valid]
                particle_mask_cpu = particle_mask_cpu[valid]

            while True:
                context_masks, target_masks = create_random_masks(
                    p4_spatial, subjets, ratio=options.trgt_ratio, max_targets=options.max_targets
                )
                real_context_counts = (context_masks & particle_mask_cpu).sum(dim=1)
                real_target_counts = (target_masks & particle_mask_cpu).sum(dim=1)
                if (real_context_counts > 0).all() and (real_target_counts > 0).all():
                    break

            context_masks = context_masks.to(device)
            target_masks = target_masks.to(device)
            context_masks_expanded = context_masks.unsqueeze(-1).expand(-1, -1, 4)
            target_masks_expanded  = target_masks.unsqueeze(-1).expand(-1, -1, 4)

            def val_step():
                cov_l = 0
                var_l = 0
                with torch.no_grad():
                    model.eval()
                    B = p4_spatial.shape[0]
                    N_ctxt = context_masks.sum(dim=1).max().item()
                    N_trgt = target_masks.sum(dim=1).max().item()
                    p4_context = p4[context_masks_expanded].view(B, N_ctxt, 4)
                    p4_target  = p4[target_masks_expanded].view(B, N_trgt, 4)
                    ctxt_particle_mask = particle_mask[context_masks].view(B, N_ctxt)
                    trgt_particle_mask = particle_mask[target_masks].view(B, N_trgt)

                    context = {"p4": p4_context, "particle_mask": ctxt_particle_mask, "split_mask": context_masks}
                    target  = {"p4": p4_target,  "particle_mask": trgt_particle_mask, "split_mask": target_masks}
                    full_jet = {"p4": p4, "p4_spatial": p4_spatial, "particle_mask": particle_mask}

                    pred_repr, target_repr, context_repr = model(context, target, full_jet, val_stats)
                    mse_loss = nn.functional.mse_loss(pred_repr, target_repr)
                    loss = mse_loss.clone()

                    if options.cov_loss_weight > 0 or options.var_loss_weight > 0:
                        context_mask_expanded = ctxt_particle_mask.unsqueeze(-1)
                        masked_context_reps = context_repr * context_mask_expanded
                        target_mask_expanded = trgt_particle_mask.unsqueeze(-1)
                        masked_target_reps = target_repr * target_mask_expanded
                        if options.cov_loss_weight > 0:
                            cov_l = (covariance_loss(target_repr) + covariance_loss(context_repr)) / 2
                            loss += options.cov_loss_weight * cov_l
                        if options.var_loss_weight > 0:
                            var_l = (variance_loss(masked_target_reps, trgt_particle_mask) +
                                     variance_loss(masked_context_reps, ctxt_particle_mask)) / 2
                            loss += options.var_loss_weight * var_l

                loss_dict = {
                    "total_loss": float(loss),
                    "mse_loss": float(mse_loss),
                    "cov_loss": float(cov_l) if options.cov_loss_weight > 0 else 0,
                    "var_loss": float(var_l) if options.var_loss_weight > 0 else 0,
                }
                return loss_dict

            val_loss_dict, etime = gpu_timer(val_step)
            loss_meter_val.update(val_loss_dict["total_loss"])
            mse_loss_meter_val.update(val_loss_dict["mse_loss"])
            if options.cov_loss_weight > 0:
                cov_loss_meter_val.update(val_loss_dict["cov_loss"])
            if options.var_loss_weight > 0:
                var_loss_meter_val.update(val_loss_dict["var_loss"])
            time_meter_val.update(etime)

            if rank == 0 and itr % options.log_freq == 0:
                logger.info(f"[{epoch + 1}, {itr}] total validation loss: {loss_meter_val.avg:.3f}, ({time_meter_val.avg:.1f} ms)")
                logger.info(f"mse loss: {mse_loss_meter_val.avg:+.3f}, cov loss: {cov_loss_meter_val.avg:+.3f}, var loss: {var_loss_meter_val.avg:+.3f}")
                log_gpu_stats(device)

        model.train()
        scheduler.step()

        if rank == 0 and epoch % options.checkpoint_freq == 0:
            save_checkpoint(
                unwrap(model),
                optimizer,
                epoch,
                loss_meter_train.avg,
                loss_meter_val.avg,
                args.output_dir,
            )

        losses_train.append(loss_meter_train.avg)
        mse_losses_train.append(mse_loss_meter_train.avg)
        if options.cov_loss_weight > 0:
            cov_losses_train.append(cov_loss_meter_train.avg)
        if options.var_loss_weight > 0:
            var_losses_train.append(var_loss_meter_train.avg)
        losses_val.append(loss_meter_val.avg)
        mse_losses_val.append(mse_loss_meter_val.avg)
        if options.cov_loss_weight > 0:
            cov_losses_val.append(cov_loss_meter_val.avg)
        if options.var_loss_weight > 0:
            var_losses_val.append(var_loss_meter_val.avg)

        if rank == 0:
            if loss_meter_val.avg < lowest_val_loss:
                logger.info(f"new lowest val loss: {loss_meter_val.avg:.3f}")
                logger.info("Saving best model")
                lowest_val_loss = loss_meter_val.avg
                torch.save(
                    unwrap(model).state_dict(),
                    os.path.join(args.output_dir, "best_model.pth"),
                )
            np.save(os.path.join(args.output_dir, "train_losses.npy"), losses_train)
            np.save(os.path.join(args.output_dir, "val_losses.npy"), losses_val)
            np.save(os.path.join(args.output_dir, "train_mse_losses.npy"), mse_losses_train)
            np.save(os.path.join(args.output_dir, "val_mse_losses.npy"), mse_losses_val)
            if options.cov_loss_weight > 0:
                np.save(os.path.join(args.output_dir, "train_cov_losses.npy"), cov_losses_train)
                np.save(os.path.join(args.output_dir, "val_cov_losses.npy"), cov_losses_val)
            if options.var_loss_weight > 0:
                np.save(os.path.join(args.output_dir, "train_var_losses.npy"), var_losses_train)
                np.save(os.path.join(args.output_dir, "val_var_losses.npy"), var_losses_val)

        epoch_end_time = time.time()
        if rank == 0:
            logger.info(f"Training time: {train_time_end - epoch_start_time:.1f} s")
            logger.info(f"Validation time: {epoch_end_time - train_time_end:.1f} s")
            logger.info(f"Epoch time:      {epoch_end_time - epoch_start_time:.1f} s")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    main(local_rank, world_size, args)

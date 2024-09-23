import os
import sys
import logging
import argparse
from pathlib import Path
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import time
from tqdm import tqdm

import torch.cuda as cuda


from src.options import Options
from src.models.jjepa import JJEPA
from src.dataset.JEPADataset import JEPADataset
from src.util.cov_loss import covariance_loss
from src.util.var_loss import variance_loss


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
    return parser.parse_args()


def setup_environment(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")


def setup_data_loader(options, data_path, world_size, rank, tag="train"):
    if tag == "val":
        data_path = data_path.replace("train", "val")
        dataset = JEPADataset(data_path, num_jets=args.num_val_jets)
    else:
        dataset = JEPADataset(data_path, num_jets=args.num_jets)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    return loader, sampler, len(dataset)


def create_random_masks(batch_size, num_subjets, device, context_scale=0.7):
    context_masks = []
    target_masks = []

    # print("Batch size", batch_size)

    for _ in range(batch_size):
        indices = torch.randperm(num_subjets, device=device)
        context_size = int(num_subjets * context_scale)
        context_indices = indices[:context_size]
        target_indices = indices[context_size:]

        context_mask = torch.zeros(num_subjets, device=device)
        target_mask = torch.zeros(num_subjets, device=device)

        context_mask[context_indices] = 1
        target_mask[target_indices] = 1

        context_masks.append(context_mask)
        target_masks.append(target_mask)

    return torch.stack(context_masks).bool(), torch.stack(target_masks).bool()


def save_checkpoint(model, optimizer, epoch, loss_train, loss_val, output_dir):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "training loss": loss_train,
        "validation loss": loss_val,
    }

    torch.save(
        checkpoint, os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
    )


# Create a logger
logger = logging.getLogger(__name__)


def setup_logging(rank, output_dir):
    log_file = Path(output_dir) / f"train_rank_{rank}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


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
        elapsed_time = (time.time() - start) * 1000  # Convert to milliseconds

    return result, elapsed_time


def log_gpu_stats(device):
    if torch.cuda.is_available():
        memory_allocated = (
            torch.cuda.memory_allocated(device) / 1024**3
        )  # Converted to GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        utilization = cuda.utilization(device)
        logger.info(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
        logger.info(f"GPU Memory Reserved: {memory_reserved:.2f} GB")
        logger.info(f"GPU Utilization: {utilization}%")


def main(rank, world_size, args):
    out_dir = args.output_dir
    if os.path.isdir(out_dir):
        # List all items in the directory
        contents = os.listdir(out_dir)

        # Filter out log files (assuming log files end with '.log')
        non_log_files = [file for file in contents if file.endswith(".pth")]

        # Check if there are files other than log files
        if non_log_files:
            sys.exit(
                "ERROR: experiment already exists and contains files other than log files; don't want to overwrite it by mistake"
            )

    # This will create the directory if it does not exist or if it is empty
    os.makedirs(out_dir, exist_ok=True)
    if world_size > 1:
        setup_environment(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    args.num_val_jets = args.num_jets // 4

    options = Options.load(args.config)
    options.batch_size = args.batch_size
    options.num_steps_per_epoch = options.num_jets // options.batch_size
    options.cov_loss_weight = args.cov_loss_weight
    options.var_loss_weight = args.var_loss_weight

    setup_logging(rank, args.output_dir)
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    logger.info(f"covariance loss weight: {options.cov_loss_weight}")
    logger.info(f"variance loss weight: {options.var_loss_weight}")

    model = JJEPA(options).to(device)
    model = model.to(dtype=torch.float32)
    if args.load_checkpoint and Path(args.load_checkpoint).is_file():
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank])

    train_loader, train_sampler, train_dataset_size = setup_data_loader(
        options, args.data_path, world_size, rank, tag="train"
    )
    val_loader, val_sampler, val_dataset_size = setup_data_loader(
        options, args.data_path, world_size, rank, tag="val"
    )
    logger.info(f"Train dataset size: {train_dataset_size}")
    logger.info(f"Val dataset size: {val_dataset_size}")

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

    logger.info("Using AdamW")
    if args.lr:
        options.lr = args.lr
    optimizer = optim.AdamW(
        param_groups,
        lr=options.lr,
        weight_decay=options.weight_decay,
        eps=options.eps,
    )
    # optimizer = optim.AdamW(
    #     model.parameters(), lr=options.lr, weight_decay=options.weight_decay
    # )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=options.num_epochs
    )
    scaler = GradScaler()

    momentum_scheduler = create_momentum_scheduler(options)

    losses_train = []
    losses_val = []
    lowest_val_loss = np.inf

    for epoch in range(options.start_epochs, options.num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        logger.info("lr: %f" % scheduler.get_last_lr()[0])

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

        pbar_t = tqdm(
            train_loader,
            total=int(train_dataset_size / options.batch_size),
            desc="Training",
        )

        for itr, (
            x,
            particle_features,
            subjets,
            particle_indices,
            subjet_mask,
            particle_mask,
        ) in enumerate(pbar_t):
            x = x.to(dtype=torch.float32)
            x = x.to(device, non_blocking=True)
            x = x.view(
                x.shape[0],
                options.num_subjets,
                options.num_particles * options.num_part_ftr,
            )

            particle_features = particle_features.to(
                device, non_blocking=True, dtype=torch.float32
            )
            subjets = subjets.to(device, non_blocking=True, dtype=torch.float32)
            particle_indices = particle_indices.to(
                device, non_blocking=True, dtype=torch.float32
            )
            subjet_mask = subjet_mask.to(device, non_blocking=True, dtype=torch.float32)
            particle_mask = particle_mask.to(
                device, non_blocking=True, dtype=torch.float32
            )

            context_masks, target_masks = create_random_masks(
                x.shape[0], options.num_subjets, device
            )

            current_momentum = next(momentum_scheduler)
            for param_group in optimizer.param_groups:
                param_group["momentum"] = current_momentum

            def train_step():
                # options = Options.load(args.config)
                optimizer.zero_grad()

                # print(" subjet", subjets.shape)
                # print("context mask", context_masks.shape)
                # print("target mask", target_masks.shape)

                # remove the zeros and collapse it to the correct shape
                sub_j_context = subjets[context_masks]
                num_ctxt_selected = context_masks.sum(
                    dim=1
                ).min()  # Minimum to handle potentially non-uniform selections
                selected_sub_j_context = sub_j_context.view(
                    subjets.shape[0], num_ctxt_selected, subjets.shape[-1]
                )
                # print("sub_j_context", selected_sub_j_context.shape)

                sub_j_target = subjets[target_masks]
                num_tgt_selected = target_masks.sum(
                    dim=1
                ).min()  # Minimum to handle potentially non-uniform selections
                selected_sub_j_target = sub_j_target.view(
                    subjets.shape[0], num_tgt_selected, subjets.shape[-1]
                )
                # print("sub_j_target", selected_sub_j_target.shape)

                context_subjets_mask = subjet_mask[context_masks]
                num_cxt_subj_mask_selected = context_masks.sum(
                    dim=1
                ).min()  # Minimum to handle potentially non-uniform selections
                context_subjets_mask = context_subjets_mask.view(
                    subjets.shape[0], num_cxt_subj_mask_selected
                )

                target_subject_mask = subjet_mask[target_masks]
                num_trg_subj_mask_selected = target_masks.sum(
                    dim=1
                ).min()  # Minimum to handle potentially non-uniform selections
                target_subjets_mask = target_subject_mask.view(
                    subjets.shape[0], num_trg_subj_mask_selected
                )

                with autocast(enabled=options.use_amp):
                    context = {
                        "subjets": selected_sub_j_context.to(device),
                        "particle_mask": particle_mask.to(device),
                        "subjet_mask": context_subjets_mask.to(device),
                        "split_mask": context_masks.to(device),
                    }
                    target = {
                        "subjets": selected_sub_j_target.to(device),
                        "particle_mask": particle_mask.to(device),
                        "subjet_mask": target_subjets_mask.to(device),
                        "split_mask": target_masks.to(device),
                    }
                    full_jet = {
                        "particles": x,
                        "particle_mask": particle_mask.to(device),
                        "subjet_mask": subjet_mask.to(device),
                        "subjets": subjets.to(device),
                    }

                    pred_repr, target_repr, context_repr = model(
                        context, target, full_jet
                    )
                    mse_loss = nn.functional.mse_loss(pred_repr, target_repr)
                    loss = mse_loss.clone()
                    if options.cov_loss_weight > 0:
                        cov_loss = (
                            covariance_loss(target_repr) / 2
                            + covariance_loss(context_repr) / 2
                        )
                        loss += options.cov_loss_weight * cov_loss
                    if options.var_loss_weight > 0:
                        var_loss = (
                            variance_loss(target_repr) / 2
                            + variance_loss(context_repr) / 2
                        )
                        loss += options.var_loss_weight * var_loss

                    if options.use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), options.max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(
                        #     model.parameters(), options.max_grad_norm
                        # )
                        optimizer.step()

                    # Step 3. momentum update of target encoder
                    with torch.no_grad():
                        m = next(momentum_scheduler)
                        for param_q, param_k in zip(
                            model.context_transformer.parameters(),
                            model.target_transformer.parameters(),
                        ):
                            param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                loss_dict = {
                    "total_loss": float(loss),
                    "mse_loss": float(mse_loss),
                    "cov_loss": float(cov_loss) if options.cov_loss_weight > 0 else 0,
                    "var_loss": float(var_loss) if options.var_loss_weight > 0 else 0,
                }
                return loss_dict

            loss_dict, etime = gpu_timer(train_step)
            loss_meter_train.update(loss_dict["total_loss"])
            mse_loss_meter_train.update(loss_dict["mse_loss"])
            cov_loss_meter_train.update(loss_dict["cov_loss"])
            var_loss_meter_train.update(loss_dict["var_loss"])
            time_meter_train.update(etime)

            if itr % options.log_freq == 0:
                logger.info(
                    f"[{epoch + 1}, {itr}] total training loss: {loss_meter_train.avg:.3f}, ({time_meter_train.avg:.1f} ms)"
                )
                logger.info(
                    f"mse loss: {mse_loss_meter_train.avg:+.3f}, cov loss: {cov_loss_meter_train.avg:+.3f}, var loss: {var_loss_meter_train.avg:+.3f}"
                )
                log_gpu_stats(device)

        # validation
        pbar_v = tqdm(
            val_loader,
            total=int(val_dataset_size / options.batch_size),
            desc="Validation",
        )

        for itr, (
            x,
            particle_features,
            subjets,
            particle_indices,
            subjet_mask,
            particle_mask,
        ) in enumerate(pbar_v):
            x = x.to(dtype=torch.float32)
            x = x.to(device, non_blocking=True)
            x = x.view(
                x.shape[0],
                options.num_subjets,
                options.num_particles * options.num_part_ftr,
            )

            particle_features = particle_features.to(
                device, non_blocking=True, dtype=torch.float32
            )
            subjets = subjets.to(device, non_blocking=True, dtype=torch.float32)
            particle_indices = particle_indices.to(
                device, non_blocking=True, dtype=torch.float32
            )
            subjet_mask = subjet_mask.to(device, non_blocking=True, dtype=torch.float32)
            particle_mask = particle_mask.to(
                device, non_blocking=True, dtype=torch.float32
            )

            context_masks, target_masks = create_random_masks(
                x.shape[0], options.num_subjets, device
            )

            def val_step():
                optimizer.zero_grad()

                # print(" subjet", subjets.shape)
                # print("context mask", context_masks.shape)
                # print("target mask", target_masks.shape)

                # remove the zeros and collapse it to the correct shape
                sub_j_context = subjets[context_masks]
                num_ctxt_selected = context_masks.sum(
                    dim=1
                ).min()  # Minimum to handle potentially non-uniform selections
                selected_sub_j_context = sub_j_context.view(
                    subjets.shape[0], num_ctxt_selected, subjets.shape[-1]
                )
                # print("sub_j_context", selected_sub_j_context.shape)

                sub_j_target = subjets[target_masks]
                num_tgt_selected = target_masks.sum(
                    dim=1
                ).min()  # Minimum to handle potentially non-uniform selections
                selected_sub_j_target = sub_j_target.view(
                    subjets.shape[0], num_tgt_selected, subjets.shape[-1]
                )
                # print("sub_j_target", selected_sub_j_target.shape)

                context_subjets_mask = subjet_mask[context_masks]
                num_cxt_subj_mask_selected = context_masks.sum(
                    dim=1
                ).min()  # Minimum to handle potentially non-uniform selections
                context_subjets_mask = context_subjets_mask.view(
                    subjets.shape[0], num_cxt_subj_mask_selected
                )

                target_subject_mask = subjet_mask[target_masks]
                num_trg_subj_mask_selected = target_masks.sum(
                    dim=1
                ).min()  # Minimum to handle potentially non-uniform selections
                target_subjets_mask = target_subject_mask.view(
                    subjets.shape[0], num_trg_subj_mask_selected
                )

                with torch.no_grad():
                    model.eval()
                    context = {
                        "subjets": selected_sub_j_context.to(device),
                        "particle_mask": particle_mask.to(device),
                        "subjet_mask": context_subjets_mask.to(device),
                        "split_mask": context_masks.to(device),
                    }
                    target = {
                        "subjets": selected_sub_j_target.to(device),
                        "particle_mask": particle_mask.to(device),
                        "subjet_mask": target_subjets_mask.to(device),
                        "split_mask": target_masks.to(device),
                    }
                    full_jet = {
                        "particles": x,
                        "particle_mask": particle_mask.to(device),
                        "subjet_mask": subjet_mask.to(device),
                        "subjets": subjets.to(device),
                    }

                    pred_repr, target_repr, context_repr = model(
                        context, target, full_jet
                    )
                    mse_loss = nn.functional.mse_loss(pred_repr, target_repr)
                    loss = mse_loss.clone()
                    if options.cov_loss_weight > 0:
                        cov_loss = (
                            covariance_loss(target_repr) / 2
                            + covariance_loss(context_repr) / 2
                        )
                        loss += options.cov_loss_weight * cov_loss
                    if options.var_loss_weight > 0:
                        var_loss = (
                            variance_loss(target_repr) / 2
                            + variance_loss(context_repr) / 2
                        )
                        loss += options.var_loss_weight * var_loss

                loss_dict = {
                    "total_loss": float(loss),
                    "mse_loss": float(mse_loss),
                    "cov_loss": float(cov_loss) if options.cov_loss_weight > 0 else 0,
                    "var_loss": float(var_loss) if options.var_loss_weight > 0 else 0,
                }
                return loss_dict

            val_loss_dict, etime = gpu_timer(val_step)
            loss_meter_val.update(val_loss_dict["total_loss"])
            mse_loss_meter_val.update(val_loss_dict["mse_loss"])
            cov_loss_meter_val.update(val_loss_dict["cov_loss"])
            var_loss_meter_val.update(val_loss_dict["var_loss"])
            time_meter_val.update(etime)

            if itr % options.log_freq == 0:
                logger.info(
                    f"[{epoch + 1}, {itr}] total validation loss: {loss_meter_val.avg:.3f}, ({time_meter_val.avg:.1f} ms)"
                )
                logger.info(
                    f"mse loss: {mse_loss_meter_val.avg:+.3f}, cov loss: {cov_loss_meter_val.avg:+.3f}, var loss: {var_loss_meter_val.avg:+.3f}"
                )
                log_gpu_stats(device)

        scheduler.step()
        if epoch % options.checkpoint_freq == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                loss_meter_train.avg,
                loss_meter_val,
                args.output_dir,
            )

        losses_train.append(loss_meter_train.avg)
        losses_val.append(loss_meter_val.avg)
        if loss_meter_val.avg < lowest_val_loss:
            logger.info(f"new lowest val loss: {loss_meter_val.avg:.3f}")
            logger.info("Saving best model")
            lowest_val_loss = loss_meter_val.avg
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "best_model.pth"),
            )
        np.save(os.path.join(args.output_dir, "train_losses.npy"), losses_train)
        np.save(os.path.join(args.output_dir, "val_losses.npy"), losses_val)


if __name__ == "__main__":
    args = parse_args()
    world_size = min(args.num_gpus, torch.cuda.device_count())
    if world_size > 1:
        torch.multiprocessing.spawn(
            main, args=(world_size, args), nprocs=world_size, join=True
        )
    else:
        main(0, 1, args)

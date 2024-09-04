import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.amp import GradScaler, autocast
import torch.distributed as dist
import time

from options import Options
from models.jjepa import JJEPA
from dataset.JEPADataset import JEPADataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train JJEPA model")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    return parser.parse_args()

def setup_environment(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

def setup_data_loader(options, data_path, world_size, rank):
    dataset = JEPADataset(data_path, num_jets=options.num_jets)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset, 
        batch_size=options.batch_size, 
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=True,
        sampler=sampler
    )
    return loader, sampler

def create_random_masks(batch_size, num_subjets, device, context_scale=0.7):
    context_masks = []
    target_masks = []

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

def save_checkpoint(model, optimizer, epoch, loss, output_dir):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

def setup_logging(rank, output_dir):
    log_file = Path(output_dir) / f"train_rank_{rank}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
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
    return cosine_scheduler(options.base_momentum, 1, options.num_epochs, options.num_steps_per_epoch)

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

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

def main(rank, world_size, args):
    if world_size > 1:
        setup_environment(rank)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    options = Options()
    options.load(args.config)
    setup_logging(rank, args.output_dir)
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    
    model = JJEPA(options).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank])

    train_loader, train_sampler = setup_data_loader(options, args.data_path, world_size, rank)

    optimizer = optim.AdamW(model.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options.num_epochs)
    scaler = GradScaler()

    momentum_scheduler = create_momentum_scheduler(options)

    for epoch in range(options.start_epochs, options.num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        if train_sampler:
            train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (x, particle_features, subjets, particle_indices, subjet_mask, particle_mask) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            particle_features = particle_features.to(device, non_blocking=True)
            subjets = subjets.to(device, non_blocking=True)
            particle_indices = particle_indices.to(device, non_blocking=True)
            subjet_mask = subjet_mask.to(device, non_blocking=True)
            particle_mask = particle_mask.to(device, non_blocking=True)
            
            context_masks, target_masks = create_random_masks(
                options.batch_size, 
                options.num_subjets,
                device
            )

            current_momentum = next(momentum_scheduler)
            for param_group in optimizer.param_groups:
                param_group['momentum'] = current_momentum

            def train_step():
                options = Options()
                options.load("src/test_options.json")
                optimizer.zero_grad()

                with autocast(enabled=options.use_amp):
                    context = {
                        'subjets': subjets * context_masks.unsqueeze(-1),
                        'particle_mask': particle_mask,
                        'subjet_mask': subjet_mask * context_masks,
                        'split_mask': context_masks,
                    }
                    target = {
                        'subjets': subjets * target_masks.unsqueeze(-1),
                        'particle_mask': particle_mask,
                        'subjet_mask': subjet_mask * target_masks,
                        'split_mask': target_masks,
                    }
                    full_jet = {
                        'particles': x,
                        'particle_mask': particle_mask,
                        'subjet_mask': subjet_mask,
                    }

                    pred_repr, target_repr = model(context, target, full_jet)
                    loss = nn.functional.mse_loss(pred_repr, target_repr)

                if options.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), options.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), options.max_grad_norm)
                    optimizer.step()

                return float(loss)

            loss, etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            if itr % options.log_freq == 0:
                logger.info(f'[{epoch + 1}, {itr}] loss: {loss_meter.avg:.3f} ({time_meter.avg:.1f} ms)')

        scheduler.step()
        save_checkpoint(model, optimizer, epoch, loss_meter.avg, args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main(0, 1, args)
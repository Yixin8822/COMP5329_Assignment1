"""
train_utils.py — Low-level training utilities used by train().
"""

import os

import numpy as np
import torch
from tqdm import tqdm


def train_single_epoch(model, optimizer, scheduler, data_iter,
                       steps, grad_clip, loss_fn, device,
                       global_step: int = 0) -> float:
    """
    Run one block of `steps` training iterations consuming from `data_iter`.
    Returns the mean loss over this block.
    """
    model.train()
    loss_list = []

    for _ in tqdm(range(steps), total=steps):
        optimizer.zero_grad(set_to_none=True)

        Cwid, Ccid, Qwid, Qcid, y1, y2, _ = next(data_iter)
        Cwid, Ccid = Cwid.to(device), Ccid.to(device)
        Qwid, Qcid = Qwid.to(device), Qcid.to(device)
        y1, y2     = y1.to(device),   y2.to(device)

        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        loss   = loss_fn(p1, p2, y1, y2)
        loss_list.append(float(loss.item()))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

    mean_loss = float(np.mean(loss_list))
    print(f"STEP {global_step + steps:8d}  loss {mean_loss:8f}\n")
    return mean_loss


def load_checkpoint(save_dir, ckpt_name, model, optimizer, scheduler, device):
    """
    Load model/optimizer/scheduler states from an existing checkpoint.
    Returns (start_step, best_f1, best_em, history) so training can resume.
    Returns (0, 0.0, 0.0, []) if no checkpoint exists.
    """
    ckpt_path = os.path.join(save_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        return 0, 0.0, 0.0, []

    print(f"Resuming from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_step = ckpt["step"]
    best_f1    = ckpt.get("best_f1", 0.0)
    best_em    = ckpt.get("best_em", 0.0)
    history    = ckpt.get("history", [])
    print(f"  Resumed at step {start_step}  best_f1={best_f1:.4f}  best_em={best_em:.4f}")
    return start_step, best_f1, best_em, history


def save_checkpoint(save_dir, ckpt_name, model, optimizer, scheduler,
                    step, best_f1, best_em, config, history=None):
    """Save model, optimizer, scheduler state to a checkpoint file."""
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step":            step,
        "best_f1":         best_f1,
        "best_em":         best_em,
        "config":          config,
        "history":         history or [],
    }
    torch.save(payload, os.path.join(save_dir, ckpt_name))

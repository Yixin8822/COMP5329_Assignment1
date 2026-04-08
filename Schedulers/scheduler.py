from functools import partial

from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR


# ── Scheduler factories ──────────────────────────────────────────────────────

def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


def _constant_lr(_):
    return 1.0


def _linear_warmup_factor(step: int, total_steps: int, warmup_ratio: float) -> float:
    if total_steps <= 0:
        return 1.0
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    return min(1.0, float(step + 1) / float(warmup_steps))


def lambda_scheduler(optimizer, args):
    """Linear warmup to the base learning rate over the first warmup steps."""
    return LambdaLR(
        optimizer,
        lr_lambda=partial(
            _linear_warmup_factor,
            total_steps=getattr(args, "num_steps", 0),
            warmup_ratio=getattr(args, "warmup_ratio", 0.1),
        ),
    )


def none_scheduler(optimizer, args):
    """No-op scheduler: learning rate never changes."""
    return LambdaLR(optimizer, lr_lambda=_constant_lr)


# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
    "none":    none_scheduler,
}

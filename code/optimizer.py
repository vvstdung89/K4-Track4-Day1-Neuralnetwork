"""optimizer.py — chọn bộ tối ưu, bộ lập lịch lr (tuỳ chọn) và cắt gradient.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")
SCHEDULERS = (None, "cosine", "warmup", "warmup_cosine")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Chú ý: weight_decay của Adam (L2 trộn vào gradient) khác weight_decay của AdamW (suy giảm tách riêng).
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải là một trong {OPTIMIZERS}, nhận {name!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, warmup_steps: int = 0, eta_min: float = 0.0):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, gọi .step() sau MỖI bước cập nhật.

    name:
        None            : không dùng (lr cố định) -> trả về None
        "cosine"        : CosineAnnealingLR, lr giảm từ lr ban đầu về eta_min sau total_steps bước
        "warmup"        : tăng tuyến tính từ ~0 lên lr trong warmup_steps bước, rồi giữ nguyên lr
                          (dùng cho quy tắc "lô ×k thì lr ×k, kèm khởi động")
        "warmup_cosine" : khởi động như trên, rồi cosine về 0
    Nếu dùng scheduler ở một thí nghiệm, ghi vào bảng (cột notes).
    """
    if name not in SCHEDULERS:
        raise ValueError(f"scheduler phải là một trong {SCHEDULERS}, nhận {name!r}")
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if name == "warmup":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    max_norm = None: chỉ đo chuẩn toàn cục, không cắt (clip_grad_norm_ với max_norm = inf không đổi gradient).
    Giá trị trả về chính là `grad_norm` phải ghi lại ở mỗi bước (để thấy "gai" gradient).
    Khi dùng mixed precision FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    limit = float("inf") if max_norm is None else float(max_norm)
    total_norm = torch.nn.utils.clip_grad_norm_(params, limit)
    return float(total_norm)

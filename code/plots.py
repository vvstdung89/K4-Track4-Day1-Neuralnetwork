"""plots.py — ảnh từng thí nghiệm và ảnh chồng.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt

MAJORITY_ACC = 0.4876   # mốc "luôn đoán lớp đa số" trên val


def _cfg_title(cfg: dict) -> str:
    hidden = "-".join(str(h) for h in cfg["hidden"])
    clip = "none" if cfg.get("clip_norm") is None else f"{cfg['clip_norm']:g}"
    line2 = (f"{cfg['loss'].upper()} | {cfg['optimizer']} lr={cfg['lr']:g} wd={cfg.get('weight_decay', 0):g} | "
             f"batch {cfg['batch']} × {cfg['epochs']} ep | {hidden} | dropout {cfg['dropout']:g} | init {cfg['init']} | "
             f"clip {clip} | {cfg['precision']} | seed {cfg['seed']}")
    if cfg.get("scheduler"):
        line2 += f" | {cfg['scheduler']}"
    return f"{cfg['exp_id']}  ({cfg['group']})\n{line2}"


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG 3 ô:
         (1) train_loss (eval mode, tập con cố định) và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch, kèm mốc "đoán đa số"
         (3) grad_norm trung bình (và lớn nhất) mỗi epoch, đo TRƯỚC khi clip; đường ngang = ngưỡng clip nếu có
    Đường đứng nét đứt = best_epoch. Tiêu đề ghi exp_id và cấu hình.
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    loss_name = cfg["loss"].upper()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label="train (eval mode)")
    ax.plot(ep, h["val_loss"], "o-", ms=3, label="val")
    ax.set_title(f"{loss_name} loss  (step 0 val = {s['step0_loss']:.3f})")
    ax.set_xlabel("epoch"); ax.set_ylabel(f"{loss_name} loss")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, label="val accuracy")
    ax.plot(ep, h["val_macro_f1"], "o-", ms=3, label="val macro-F1")
    ax.axhline(MAJORITY_ACC, color="gray", ls=":", lw=1, label="đoán đa số (acc)")
    ax.set_title(f"val acc {s['val_acc']:.4f} | macro-F1 {s['val_macro_f1']:.4f} (best epoch)")
    ax.set_xlabel("epoch"); ax.set_ylabel("điểm")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, label="trung bình mỗi epoch")
    ax.plot(ep, h["grad_norm_max"], "--", lw=1, label="lớn nhất mỗi epoch")
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], color="red", ls=":", lw=1.2, label=f"ngưỡng clip c = {cfg['clip_norm']:g}")
    finite = [v for v in h["grad_norm_max"] if math.isfinite(v) and v > 0]
    if finite and max(finite) / min(finite) > 30:
        ax.set_yscale("log")
    ax.set_title("grad norm L2 toàn cục (trước clip)")
    ax.set_xlabel("epoch"); ax.set_ylabel("‖g‖")

    for ax in axes:
        if s.get("best_epoch"):
            ax.axvline(s["best_epoch"], color="k", ls="--", lw=0.8, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    title = _cfg_title(cfg) + ("   [DIVERGED]" if s.get("diverged") else "")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "") -> None:
    """Vẽ chồng một (hoặc vài) chỉ số của nhiều thí nghiệm, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    metric: tên một khoá của history ("val_loss", "val_macro_f1", "grad_norm", ...) hoặc list các khoá
            (mỗi khoá một ô). Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for r in results:
            ax.plot(r["history"]["epoch"], r["history"][m], "o-", ms=2.5, lw=1.3, label=r["cfg"]["exp_id"])
        ax.set_xlabel("epoch"); ax.set_ylabel(m); ax.set_title(m)
        if m.startswith("grad_norm"):
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

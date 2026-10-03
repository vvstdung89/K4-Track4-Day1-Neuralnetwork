"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7
TRAIN_EVAL_SIZE = 50_000   # train_loss đo ở eval mode trên một tập con CỐ ĐỊNH (giống nhau cho mọi thí nghiệm)
TRAIN_EVAL_SEED = 0

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" | "warmup_cosine" (tuỳ chọn; ghi vào notes nếu dùng)
    warmup_steps=0,
    seed=1,
    notes="",
)

AMP_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_class_scores(cm: np.ndarray):
    """precision, recall, F1 của từng lớp từ ma trận nhầm lẫn (hàng = thật, cột = dự đoán); 0 khi mẫu số = 0."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    return float(per_class_scores(cm)[2].mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval()."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    loss là trung bình trên toàn bộ X (cộng dồn loss × kích thước lô rồi chia N), cùng định nghĩa với loss lúc train.
    Dùng hàm này cho: train loss (tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    n = len(X)
    total = torch.zeros((), dtype=torch.float64, device=X.device)
    cm = torch.zeros(N_CLASSES * N_CLASSES, dtype=torch.int64, device=X.device)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb).float()
        total += compute_loss(logits, yb, loss_name).double() * len(yb)
        cm += torch.bincount(yb * N_CLASSES + logits.argmax(dim=1), minlength=N_CLASSES * N_CLASSES)
    cm = cm.view(N_CLASSES, N_CLASSES).cpu().numpy()
    return dict(loss=float(total) / n, acc=float(np.trace(cm)) / n, macro_f1=macro_f1_from_confusion(cm))


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit thô và one-hot của y, lấy trung bình trên MỌI phần tử (B × 7) và không có
               hệ số 1/2, đúng như nn.MSELoss(reduction="mean").
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        return F.mse_loss(logits, F.one_hot(y, logits.shape[1]).to(logits.dtype))
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


def _fmt_cfg(cfg: dict) -> str:
    hidden = "-".join(str(h) for h in cfg["hidden"])
    clip = "none" if cfg["clip_norm"] is None else f"{cfg['clip_norm']:g}"
    s = (f"{cfg['loss']} | {cfg['optimizer']} lr={cfg['lr']:g} wd={cfg['weight_decay']:g} | batch {cfg['batch']} | "
         f"{hidden} drop={cfg['dropout']:g} init={cfg['init']} | clip={clip} | {cfg['precision']} | seed {cfg['seed']}")
    if cfg.get("scheduler"):
        s += f" | sched={cfg['scheduler']}"
    return s


def run_experiment(cfg: dict, data: dict, verbose: bool = True, print_every: int = 5) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (khoá thiếu lấy từ DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val trên device). X_eval KHÔNG được dùng ở đây.

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "grad_norm_max",
                     "clip_frac", "amp_skipped", "lr", "epoch_time_s"}  (mỗi khoá là list theo epoch),
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged",
                     "grad_norm_p50/p90/p99/max" (phân phối grad_norm theo bước, để chọn ngưỡng clip), ...},
         "best_state": bản sao state_dict ở epoch có val_loss thấp nhất (chỉ trong RAM, không ghi ra JSON),
         "grad_norm_steps": chuẩn gradient (trước clip) của từng bước (chỉ trong RAM; dùng để chọn ngưỡng clip)}

    Quy ước đo:
      - grad_norm: chuẩn L2 toàn cục TRƯỚC khi clip. Với fp16 luôn gọi scaler.unscale_ trước khi đo, nên giá trị
        không bị nhân với hệ số scale. Bước có gradient inf/NaN (GradScaler bỏ qua) không tính vào trung bình,
        được đếm ở "amp_skipped".
      - epoch_time_s: chỉ vòng cập nhật tham số (không gồm phần đánh giá cuối epoch), có cuda.synchronize().
      - peak_mem_MB: torch.cuda.max_memory_allocated()/2**20 trong cả lần chạy (gồm cả dữ liệu đã nằm trên GPU);
        base_mem_MB là bộ nhớ đã cấp phát lúc bắt đầu (chủ yếu là dữ liệu), để so phần tăng thêm.
      - val_acc, val_macro_f1 lấy ở best_epoch (epoch có val_loss thấp nhất), như dừng sớm.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    assert cfg["lr"] is not None, "chọn lr bằng val trước khi chạy"
    if cfg["precision"] not in AMP_DTYPES:
        raise ValueError(f"precision phải là một trong {tuple(AMP_DTYPES)}, nhận {cfg['precision']!r}")
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    on_cuda = device.type == "cuda"

    # 0. seed, model, optimizer, scaler
    set_seed(cfg["seed"])
    model = MLP(cfg["hidden"], cfg["dropout"], cfg["init"]).to(device)
    if cfg["hidden"] in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[cfg["hidden"]], "số tham số lệch bảng quy định"
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"], cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], steps_per_epoch * cfg["epochs"], cfg["warmup_steps"])
    amp_dtype = AMP_DTYPES[cfg["precision"]]
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["precision"] == "fp16")
    batch_gen = torch.Generator().manual_seed(cfg["seed"])            # thứ tự lô chỉ phụ thuộc seed
    sub = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(TRAIN_EVAL_SEED))[:TRAIN_EVAL_SIZE]
    sub = sub.to(device)
    X_sub, y_sub = X_tr[sub], y_tr[sub]

    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device) / 2**20 if on_cuda else float("nan")

    # 1. loss bước 0, TRƯỚC bước cập nhật đầu tiên
    step0_loss = evaluate(model, X_val, y_val, cfg["loss"])["loss"]

    keys = ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "grad_norm_max",
            "clip_frac", "amp_skipped", "lr", "epoch_time_s")
    hist = {k: [] for k in keys}
    grad_norm_steps = []
    best_val, best_epoch, best_state, diverged = float("inf"), None, None, False
    if verbose:
        print(f"[{cfg['exp_id']}] {_fmt_cfg(cfg)} | step0 val loss {step0_loss:.4f}")

    # 2. huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if on_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        norms, n_clipped, n_skipped = [], 0, 0
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=batch_gen):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)            # mọi bước (no-op khi không dùng fp16), để grad_norm đúng thang đo
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])
            scaler.step(optimizer)                 # = optimizer.step(), hoặc bỏ qua bước nếu gradient fp16 bị inf/NaN
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            grad_norm_steps.append(gn)
            if math.isfinite(gn):
                norms.append(gn)
                n_clipped += cfg["clip_norm"] is not None and gn > cfg["clip_norm"]
            else:
                n_skipped += 1
            if not math.isfinite(loss.item()):
                diverged = True                    # dừng sớm, không để notebook chạy tiếp với tham số NaN
                break
        if on_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        # cuối epoch: đo ở chế độ eval (dropout tắt)
        tr = evaluate(model, X_sub, y_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        for k, v in (("epoch", epoch), ("train_loss", tr["loss"]), ("val_loss", va["loss"]), ("val_acc", va["acc"]),
                     ("val_macro_f1", va["macro_f1"]),
                     ("grad_norm", float(np.mean(norms)) if norms else float("nan")),
                     ("grad_norm_max", float(np.max(norms)) if norms else float("nan")),
                     ("clip_frac", n_clipped / max(1, len(norms))), ("amp_skipped", n_skipped),
                     ("lr", optimizer.param_groups[0]["lr"]), ("epoch_time_s", epoch_time)):
            hist[k].append(v)
        if va["loss"] < best_val:                   # NaN không bao giờ < best_val
            best_val, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}   # bản sao, không phải tham chiếu
        if verbose and (epoch == 1 or epoch % print_every == 0 or epoch == cfg["epochs"] or diverged):
            print(f"  ep {epoch:>2}/{cfg['epochs']} | train {tr['loss']:.4f} | val {va['loss']:.4f} "
                  f"acc {va['acc']:.4f} f1 {va['macro_f1']:.4f} | gn {hist['grad_norm'][-1]:.3f} "
                  f"(max {hist['grad_norm_max'][-1]:.3f}) | {epoch_time:.2f}s" + ("  -> DIVERGED" if diverged else ""))
        if diverged:
            break

    # 3. tóm tắt tại best_epoch
    i = best_epoch - 1 if best_epoch is not None else len(hist["epoch"]) - 1
    finite_gn = [g for g in grad_norm_steps if math.isfinite(g)]
    gn_pct = np.percentile(finite_gn, [50, 90, 99]) if finite_gn else [float("nan")] * 3
    summary = dict(
        step0_loss=step0_loss,
        best_val_loss=best_val if best_epoch is not None else float("nan"),
        best_epoch=best_epoch,
        final_train_loss=hist["train_loss"][-1],
        final_val_loss=hist["val_loss"][-1],
        val_acc=hist["val_acc"][i],
        val_macro_f1=hist["val_macro_f1"][i],
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])),
        peak_mem_MB=torch.cuda.max_memory_allocated(device) / 2**20 if on_cuda else float("nan"),
        base_mem_MB=base_mem,
        diverged=diverged,
        grad_norm_p50=float(gn_pct[0]), grad_norm_p90=float(gn_pct[1]), grad_norm_p99=float(gn_pct[2]),
        grad_norm_max=float(max(finite_gn)) if finite_gn else float("nan"),
        epochs_run=len(hist["epoch"]),
        steps_per_epoch=steps_per_epoch,
        device=torch.cuda.get_device_name(device) if on_cuda else device.type,
    )
    if verbose:
        print(f"  -> best epoch {best_epoch} | best val loss {summary['best_val_loss']:.4f} | "
              f"val acc {summary['val_acc']:.4f} | val macro-F1 {summary['val_macro_f1']:.4f} | "
              f"{summary['time_per_epoch_s']:.2f}s/epoch" + (" | DIVERGED" if diverged else ""))
    return dict(cfg=cfg, history=hist, summary=summary, best_state=best_state, grad_norm_steps=grad_norm_steps)


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    raise NotImplementedError  # TODO


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    raise NotImplementedError  # TODO

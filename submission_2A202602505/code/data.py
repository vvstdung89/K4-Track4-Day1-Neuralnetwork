"""data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị, chia lô.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10   # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def _check_xy(X, y, name: str) -> None:
    assert X.dtype == np.float32 and X.ndim == 2 and X.shape[1] == N_FEATURES, \
        f"{name}: X phải float32 (N, {N_FEATURES}), hiện là {X.dtype} {X.shape}"
    assert y.dtype == np.int64 and y.shape == (len(X),), \
        f"{name}: y phải int64 ({len(X)},), hiện là {y.dtype} {y.shape}"
    assert y.min() >= 0 and y.max() <= N_CLASSES - 1, f"{name}: nhãn phải nằm trong 0..{N_CLASSES - 1}"


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    with np.load(f"{processed_dir}/train.npz") as tr:
        X_train_full, y_train_full = tr["X"], tr["y"]
    with np.load(f"{processed_dir}/eval.npz") as ev:
        X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]

    _check_xy(X_train_full, y_train_full, "train")
    _check_xy(X_eval, y_eval, "eval")
    assert eval_row_id.shape == (len(X_eval),) and len(np.unique(eval_row_id)) == len(X_eval), \
        "eval_row_id phải có đúng một giá trị cho mỗi dòng eval, không trùng"
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Vì sao không tính trên toàn bộ dữ liệu hay trên eval: mean/std khi đó mang thông tin của chính
    các mẫu dùng để đo (rò rỉ), nên điểm val/eval không còn phản ánh dữ liệu mới thật sự.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)   # cộng dồn bằng float64 cho chính xác
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    std[std == 0] = 1.0                            # cột hằng: giữ nguyên thay vì chia cho 0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    out = X.copy()
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / std
    return out


def _class_pct(y) -> np.ndarray:
    return 100 * np.bincount(y, minlength=N_CLASSES) / len(y)


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id, mean, std (thống kê chuẩn hoá, chỉ tính trên X_tr)
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)

    mean, std = fit_standardizer(X_tr)             # chỉ X_tr: val và eval không góp vào thống kê
    X_tr, X_val, X_eval = (apply_standardizer(X, mean, std) for X in (X_tr, X_val, X_eval))

    print(f"train_full {len(X_full):,} -> X_tr {X_tr.shape}, X_val {X_val.shape} | X_eval {X_eval.shape}")
    print("lớp   tr(%)   val(%)  eval(%)")
    for c, (a, b, e) in enumerate(zip(_class_pct(y_tr), _class_pct(y_val), _class_pct(y_eval))):
        print(f"{c:>3} {a:7.3f} {b:8.3f} {e:8.3f}")
    majority = int(np.bincount(y_tr, minlength=N_CLASSES).argmax())   # lớp đa số lấy từ train, không từ val
    print(f"luôn đoán lớp đa số ({majority}) -> accuracy trên val = {(y_val == majority).mean():.4f}")

    def to_dev(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device)

    return dict(
        X_tr=to_dev(X_tr), y_tr=to_dev(y_tr),
        X_val=to_dev(X_val), y_val=to_dev(y_val),
        X_eval=to_dev(X_eval), y_eval=to_dev(y_eval),
        eval_row_id=eval_row_id, mean=mean, std=std,
    )


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True,
                    drop_last: bool = False):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Hoán vị được sinh trên thiết bị của `generator` (CPU nếu không truyền) rồi chuyển sang X.device,
    nên một torch.Generator CPU vẫn dùng được khi dữ liệu nằm trên GPU, và thứ tự lô chỉ phụ thuộc seed.
    Batch cuối: mặc định GIỮ lô cuối nhỏ hơn (drop_last=False), để mỗi epoch dùng đủ mọi mẫu.
    Với X_tr = 371 847 mẫu và batch 512, lô cuối có 135 mẫu (1/727 bước), ảnh hưởng không đáng kể.
    """
    n = len(X)
    if shuffle:
        gen_device = generator.device if generator is not None else "cpu"
        idx_all = torch.randperm(n, generator=generator, device=gen_device).to(X.device)
    else:
        idx_all = torch.arange(n, device=X.device)
    stop = n - n % batch_size if drop_last else n
    for i in range(0, stop, batch_size):
        idx = idx_all[i:i + batch_size]
        yield X[idx], y[idx]

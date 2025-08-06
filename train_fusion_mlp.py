#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fusion_mlp.py

Fit an MLP to predict composite execution cost (ms) from:
  [room context (5) || IMC composite vector (6)].

Inputs
------
- --dataset_root: path to LeRobotDataset episodes written by select_and_execute.py
- --exec_index:   executed_index.json emitted by select_and_execute.py
- --imc_vectors:  imc_out/composites.npy (N x 6)
- --alpha, --beta: label = time_ms + alpha*energy_mWh + beta*[safety_incident]
- (optional) --use_log_targets: train on log1p(cost_ms), report real-ms metrics

Outputs (in --out_dir)
----------------------
- best_model.pt     : best val RMSE checkpoint (state_dict + metadata)
- last_model.pt     : last epoch checkpoint
- scalers.npz       : feature/target scalers (mu/scale)
- config.json       : run configuration
- metrics.json      : final metrics (train/val MAE/RMSE in ms)

Data alignment
--------------
Relies on executed_index.json of the form:
{
  "room_000": [{"pid": 0, "imc_idx": 123, "plan_key": "..."}],
  "room_001": [...],
  ...
}

Each episode parquet must contain rows with:
  action == "composite_summary" and plan_id == pid.
We take that row's time_ms, energy_mWh, safety_incident, and the episode's
room context (first/meta row) to build training samples.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Utilities
# ---------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def choose_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)

def find_parquets(root: Path) -> List[Path]:
    return sorted(root.rglob("*.parquet"))

def read_exec_index(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    return json.loads(Path(path).read_text())

def read_context_row(df: pd.DataFrame) -> np.ndarray:
    # Meta row first (as in your logger); fallback safely
    size    = float(df["room_size"].iloc[0])    if "room_size"    in df.columns else 20.0
    clutter = float(df["room_clutter"].iloc[0]) if "room_clutter" in df.columns else 0.5
    heat    = float(df["room_heat"].iloc[0])    if "room_heat"    in df.columns else 0.0
    gas     = float(df["room_gas"].iloc[0])     if "room_gas"     in df.columns else 0.0
    noise   = float(df["room_noise"].iloc[0])   if "room_noise"   in df.columns else 0.0
    return np.array([size, clutter, heat, gas, noise], dtype=np.float32)

def compute_label_ms(time_ms: float, energy_mWh: float, safety_incident: bool,
                     alpha: float, beta: float) -> float:
    return float(time_ms) + float(alpha) * float(energy_mWh) + (float(beta) if bool(safety_incident) else 0.0)

# ---------------------------
# Dataset assembly
# ---------------------------

def build_samples(dataset_root: Path,
                  exec_index_path: Path,
                  imc_vectors_path: Path,
                  alpha: float,
                  beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X  : [num_samples, D] where D = 5 (context) + 6 (IMC)
      y  : [num_samples, 1] cost in ms
    """
    # Load IMC vectors
    C = np.load(imc_vectors_path)  # (N, 6)
    if C.ndim != 2:
        raise ValueError(f"IMC vectors must be 2D. Got {C.shape}")
    if C.shape[1] != 6:
        # Allow different dim but warn (update FEATURE_ORDER in your pipeline if needed)
        print(f"[warn] Expected 6-dim IMC vectors, got {C.shape[1]}. Proceeding.")

    # Load executed index
    idx = read_exec_index(exec_index_path)
    # Discover parquets
    shards = find_parquets(dataset_root)
    if not shards:
        raise SystemExit(f"❌ No .parquet files under {dataset_root}")
    # Map room key to shard path by suffix match (room_{r:03d})
    # We assume one parquet per episode and that ordering matches room numbering.
    roomkey_to_path: Dict[str, Path] = {}
    # Create a deterministic order from filename sort
    for p in shards:
        # Accept any file; actual mapping uses order of keys in exec_index
        pass

    # Construct samples
    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []

    # We'll iterate rooms in sorted room keys order, and align by position with sorted shard list.
    room_keys = sorted(idx.keys())  # e.g., ["room_000", ...]
    if len(room_keys) > len(shards):
        print(f"[warn] exec_index has {len(room_keys)} rooms but only {len(shards)} parquets; truncating to min().")
    use_n = min(len(room_keys), len(shards))
    room_keys = room_keys[:use_n]
    shards = shards[:use_n]

    for r_i, (room_key, shard_path) in enumerate(zip(room_keys, shards)):
        df = pd.read_parquet(shard_path)
        ctx = read_context_row(df)  # (5,)
        # Composite summary rows in this episode
        if "action" not in df.columns or "plan_id" not in df.columns:
            print(f"[warn] Missing action/plan_id in {shard_path.name}; skipping room {room_key}.")
            continue
        mask_sum = (df["action"] == "composite_summary")
        if not mask_sum.any():
            print(f"[warn] No composite_summary rows in {shard_path.name}; skipping room {room_key}.")
            continue
        df_sum = df.loc[mask_sum, ["plan_id", "time_ms", "energy_mWh", "safety_incident"]].copy()

        # Executed plans listed in exec_index
        plan_list = idx.get(room_key, [])
        if not plan_list:
            print(f"[warn] No executed plans in exec_index for {room_key}; skipping.")
            continue

        for rec in plan_list:
            pid = int(rec["pid"])
            imc_idx = int(rec["imc_idx"])
            if imc_idx < 0 or imc_idx >= C.shape[0]:
                print(f"[warn] imc_idx {imc_idx} out of range; skipping.")
                continue
            sub = df_sum[df_sum["plan_id"] == pid]
            if sub.empty:
                print(f"[warn] No summary row for plan_id={pid} in {shard_path.name}; skipping this plan.")
                continue
            row = sub.iloc[0]
            t_ms  = float(row["time_ms"])
            e_mWh = float(row["energy_mWh"])
            safe  = bool(row["safety_incident"])
            y_ms  = compute_label_ms(t_ms, e_mWh, safe, alpha, beta)

            vec = np.concatenate([ctx.astype(np.float32), C[imc_idx].astype(np.float32)], axis=0)
            X_rows.append(vec)
            y_rows.append(y_ms)

    if not X_rows:
        raise SystemExit("❌ No training samples assembled. Check exec_index, parquet shards, and IMC vectors.")

    X = np.stack(X_rows, axis=0)
    y = np.array(y_rows, dtype=np.float32).reshape(-1, 1)
    print(f"[data] Built dataset: X={X.shape}, y={y.shape}  (ctx=5, imc={X.shape[1]-5})")
    return X, y

# ---------------------------
# PyTorch model & training
# ---------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self) -> int:
        return self.X.shape[0]
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

def split_indices(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
    return tr_idx, val_idx

def train_loop(model: nn.Module,
               opt: torch.optim.Optimizer,
               train_loader: DataLoader,
               val_loader: DataLoader,
               *,
               device: torch.device,
               epochs: int,
               save_dir: Path,
               use_log_targets: bool) -> Dict[str, Any]:
    save_dir.mkdir(parents=True, exist_ok=True)
    best = {"epoch": 0, "val_rmse_ms": float("inf"), "val_mae_ms": float("inf")}
    scaler_path = save_dir / "scalers.npz"   # created outside
    config_path = save_dir / "config.json"   # created outside

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.L1Loss()(pred, yb)  # MAE in (scaled space if use_log_targets else z-score space)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        # Validation (compute MAE/RMSE in *ms space*)
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                y_true.append(yb.cpu().numpy())
                y_pred.append(out.cpu().numpy())
        y_true = np.concatenate(y_true, axis=0).reshape(-1, 1)
        y_pred = np.concatenate(y_pred, axis=0).reshape(-1, 1)

        # Load scalers to invert transform (created outside)
        sc = np.load(scaler_path)
        mu_y, sc_y = sc["mu_y"], sc["sc_y"]
        # Inverse target scaling
        if use_log_targets:
            # y was scaled after log1p; invert: y_ms = expm1( y_scaled*sc_y + mu_y )
            y_pred_ms = np.expm1(y_pred * sc_y + mu_y)
            y_true_ms = np.expm1(y_true * sc_y + mu_y)
        else:
            # y was scaled from raw ms; invert: y_ms = y_scaled*sc_y + mu_y
            y_pred_ms = y_pred * sc_y + mu_y
            y_true_ms = y_true * sc_y + mu_y

        diff = y_pred_ms - y_true_ms
        val_mae = float(np.mean(np.abs(diff)))
        val_rmse = float(np.sqrt(np.mean(diff**2)))

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        print(f"epoch {ep:03d}  train_MAE={tr_loss:.4f}  val_MAE(ms)={val_mae:.1f}  val_RMSE(ms)={val_rmse:.1f}")

        # Save "last" checkpoint regardless
        last_path = save_dir / "last_model.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "epoch": ep,
        }, last_path)

        # Save "best" whenever improved (continually)
        if val_rmse < best["val_rmse_ms"]:
            best.update({"epoch": ep, "val_rmse_ms": val_rmse, "val_mae_ms": val_mae})
            best_path = save_dir / "best_model.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": ep,
                "best_metrics": best,
            }, best_path)
            print(f"✓ saved best @ epoch {ep:03d}  val_RMSE(ms)={val_rmse:.1f} → {best_path}")

    return best

# ---------------------------
# CLI & main
# ---------------------------

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a fusion MLP on context + IMC vectors to predict cost (ms).")
    ap.add_argument("--dataset_root", type=Path, required=True, help="Path to lerobot_exec_ds")
    ap.add_argument("--exec_index",   type=Path, required=True, help="Path to executed_index.json")
    ap.add_argument("--imc_vectors",  type=Path, required=True, help="Path to imc_out/composites.npy")
    ap.add_argument("--out_dir",      type=Path, default=Path("fusion_runs"))

    ap.add_argument("--alpha", type=float, default=1.0, help="ms per mWh in label")
    ap.add_argument("--beta",  type=float, default=2000.0, help="ms penalty for safety incident in label")

    ap.add_argument("--hidden", type=str, default="64,64", help="Comma-separated hidden sizes, e.g., '64,64'")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--wd",     type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument("--use_log_targets", action="store_true",
                    help="Train on log1p(cost_ms) (targets are standardized after log).")
    return ap

def main():
    args = build_cli().parse_args()
    set_seeds(args.seed)
    device = choose_device(args.device)

    X, y_ms = build_samples(args.dataset_root, args.exec_index, args.imc_vectors,
                            alpha=args.alpha, beta=args.beta)

    # Transform targets if requested
    if args.use_log_targets:
        y_for_scaling = np.log1p(y_ms.copy())
    else:
        y_for_scaling = y_ms.copy()

    # Standardize X and y (z-score)
    x_scaler = StandardScaler(with_mean=True, with_std=True)
    y_scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = x_scaler.fit_transform(X.astype(np.float32))
    yz = y_scaler.fit_transform(y_for_scaling.astype(np.float32))

    # Persist scalers for inference
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "scalers.npz",
             mu_x=x_scaler.mean_.astype(np.float32),
             sc_x=x_scaler.scale_.astype(np.float32),
             mu_y=y_scaler.mean_.astype(np.float32),
             sc_y=y_scaler.scale_.astype(np.float32),
             use_log_targets=np.array([1 if args.use_log_targets else 0], dtype=np.int64))

    # Train/val split
    n = Xz.shape[0]
    tr_idx, val_idx = split_indices(n, args.val_frac, args.seed)
    tr_ds = ArrayDataset(Xz[tr_idx], yz[tr_idx])
    va_ds = ArrayDataset(Xz[val_idx], yz[val_idx])
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    in_dim = Xz.shape[1]
    hidden = [int(s) for s in args.hidden.split(",") if s.strip()]
    model = MLP(in_dim, hidden, out_dim=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Save config
    (args.out_dir / "config.json").write_text(json.dumps({
        "in_dim": in_dim,
        "hidden": hidden,
        "alpha": args.alpha,
        "beta": args.beta,
        "use_log_targets": bool(args.use_log_targets),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "wd": args.wd,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "device": str(device),
    }, indent=2))

    best = train_loop(model, opt, tr_ld, va_ld,
                      device=device, epochs=args.epochs,
                      save_dir=args.out_dir, use_log_targets=args.use_log_targets)

    # Final metrics on both splits (in ms)
    def eval_loader(loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        yp, yt = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                out = model(xb).cpu().numpy()
                yp.append(out); yt.append(yb.numpy())
        yp = np.concatenate(yp, 0); yt = np.concatenate(yt, 0)
        sc = np.load(args.out_dir / "scalers.npz")
        mu_y, sc_y = sc["mu_y"], sc["sc_y"]
        if args.use_log_targets:
            yp_ms = np.expm1(yp * sc_y + mu_y)
            yt_ms = np.expm1(yt * sc_y + mu_y)
        else:
            yp_ms = yp * sc_y + mu_y
            yt_ms = yt * sc_y + mu_y
        diff = yp_ms - yt_ms
        mae = float(np.mean(np.abs(diff))); rmse = float(np.sqrt(np.mean(diff**2)))
        return mae, rmse

    tr_mae, tr_rmse = eval_loader(tr_ld)
    va_mae, va_rmse = eval_loader(va_ld)

    (args.out_dir / "metrics.json").write_text(json.dumps({
        "train_MAE_ms": tr_mae, "train_RMSE_ms": tr_rmse,
        "val_MAE_ms": va_mae,   "val_RMSE_ms": va_rmse,
        "best": best,
        "n_train": len(tr_ds), "n_val": len(va_ds),
    }, indent=2))

    print(f"Done. train MAE={tr_mae:.1f} RMSE={tr_rmse:.1f} | val MAE={va_mae:.1f} RMSE={va_rmse:.1f}")
    print(f"Best (during training): epoch {best['epoch']}  val_RMSE(ms)={best['val_rmse_ms']:.1f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import numpy as np
import torch
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # <-- prevent remote HF lookups, force local only

from pathlib import Path
from scipy.stats import spearmanr

# Reuse parts of your loader; minimal duplication
import argparse
from typing import Tuple, Dict, Any
import torch.nn as nn

# SimpleMLP same as in runtime_loop.py
class SimpleMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def _standardize(x, mean, std):
    if mean is None or std is None:
        return x
    m = np.asarray(mean, dtype=np.float32)
    s = np.asarray(std, dtype=np.float32)
    s = np.where(s <= 1e-8, 1.0, s)
    return (x - m) / s

def _destandardize(y, mean, std):
    if mean is None or std is None:
        return y
    # safe broadcast in case mean/std are arrays of shape (1,)
    return y * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)

def load_model_and_scaler(fusion_dir: Path, device):
    ckpt = torch.load(fusion_dir / "best_model.pt", map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    # infer dims
    def infer_dims(sd):
        pairs = []
        for k,v in sd.items():
            if k.endswith(".weight") and v.ndim == 2:
                try:
                    idx = int(k.split(".")[1])
                except:
                    idx = 9999
                pairs.append((idx, v))
        pairs.sort(key=lambda x: x[0])
        dims = [int(pairs[0][1].shape[1])]
        for _,W in pairs:
            dims.append(int(W.shape[0]))
        return dims
    dims = None
    cfg_path = fusion_dir / "model_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            if "dims" in cfg:
                dims = [int(x) for x in cfg["dims"]]
        except:
            pass
    if dims is None:
        dims = infer_dims(state)
    model = SimpleMLP(dims)
    model.load_state_dict(state)
    model.to(device).eval()

    scaler = {"x_mean": None, "x_std": None, "y_mean": None, "y_std": None, "use_log_targets": False}
    npz_path = None
    if (fusion_dir / "scalers.npz").exists():
        npz_path = fusion_dir / "scalers.npz"
    elif (fusion_dir / "scaler.npz").exists():
        npz_path = fusion_dir / "scaler.npz"
    if npz_path:
        arr = np.load(npz_path)
        if "mu_x" in arr and "sc_x" in arr:
            scaler["x_mean"] = arr["mu_x"].tolist()
            scaler["x_std"]  = arr["sc_x"].tolist()
            scaler["y_mean"] = arr["mu_y"].item() if "mu_y" in arr else None
            scaler["y_std"]  = arr["sc_y"].item() if "sc_y" in arr else None
            scaler["use_log_targets"] = bool(arr.get("use_log_targets", [0]).item())
        else:
            scaler["x_mean"] = arr["x_mean"].tolist() if "x_mean" in arr else None
            scaler["x_std"]  = arr["x_std"].tolist() if "x_std" in arr else None
            scaler["y_mean"] = arr["y_mean"].item() if "y_mean" in arr else None
            scaler["y_std"]  = arr["y_std"].item() if "y_std" in arr else None
    return model, scaler

def score(model, scaler, ctx, imc, device):
    N = imc.shape[0]
    X = np.concatenate([np.repeat(ctx[None,:], N, axis=0), imc], axis=1).astype(np.float32)
    Xs = _standardize(X, scaler.get("x_mean"), scaler.get("x_std")).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(Xs).to(device)
        yh = model(xt).squeeze(-1).cpu().numpy()
    y_ms = _destandardize(yh, scaler.get("y_mean"), scaler.get("y_std"))
    if scaler.get("use_log_targets", False):
        y_ms = np.expm1(y_ms)
    return y_ms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_dir", required=True, type=Path)
    parser.add_argument("--runtime_ds", required=True, type=Path)
    parser.add_argument("--imc_vectors", required=True, type=Path)
    parser.add_argument("--good_json", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    # normalize device (mirror runtime_loop behavior)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # load what was executed
    sel = json.loads((args.runtime_ds / "runtime_selection.json").read_text())
    executed = json.loads((args.runtime_ds / "executed_index.json").read_text())
    imc = np.load(args.imc_vectors).astype(np.float32)

    model, scaler = load_model_and_scaler(args.fusion_dir, torch_device)
    from runtime_loop import RoomProfile, _compose_ctx_from_room  # assumes runtime_loop.py is importable

    preds = []
    actuals = []

    # Load actual composite summary times directly from the local dataset parquet
    import glob
    import pandas as pd
    actual_map = {}  # (room_idx, plan_id) -> time_ms
    # Each episode corresponds to one room index in order; we rely on the fact that runtime_loop wrote rooms sequentially.
    parquet_paths = sorted(glob.glob(str(Path(args.runtime_ds) / "**" / "*.parquet"), recursive=True))
    room_idx = 0
    for ep_path in parquet_paths:
        try:
            df = pd.read_parquet(ep_path)
        except Exception:
            continue
        # filter composite summary rows
        summaries = df[df["is_summary"] == True]
        # within an episode there are multiple selected plans; plan_id distinguishes them
        for _, row in summaries.iterrows():
            pid = int(row["plan_id"])
            time_ms = float(row["time_ms"])
            actual_map[(room_idx, pid)] = time_ms
        room_idx += 1
    if not actual_map:
        print("[warn] actual_map empty; falling back to uniform split per plan", flush=True)

    for room_entry in sel["selections"]:
        r = room_entry["room"]
        ctx = np.array(room_entry["ctx"], dtype=np.float32)
        chosen = room_entry["chosen"]  # list in order => plan_id order
        # Build predictions for each selected index
        selected_indices = [c["index"] for c in chosen]
        y_ms = score(model, scaler, ctx, imc[selected_indices], torch_device)
        preds.extend(y_ms.tolist())

        # Get actuals: prefer extracted composite summary times per plan, else uniform split
        for pid in range(len(chosen)):
            actual = actual_map.get((r, pid), room_entry["episode_time_ms"] / len(chosen))
            actuals.append(actual)

    preds = np.array(preds, dtype=np.float32)
    actuals = np.array(actuals, dtype=np.float32)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(actuals, preds)
    rmse = mean_squared_error(actuals, preds, squared=False)
    rho, p = spearmanr(actuals, preds)

    print(f"Count: {len(preds)}")
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, Spearman ρ: {rho:.3f} (p={p:.3g})")
    # Optionally: residual analysis
    diffs = preds - actuals
    print(f"Predicted mean / actual mean: {preds.mean():.1f} / {actuals.mean():.1f}")
    print(f"Residuals std: {diffs.std():.1f}")
if __name__ == "__main__":
    main()

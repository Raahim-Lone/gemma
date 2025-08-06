#!/usr/bin/env python3
# build_cost_matrix.py — robust reader for LeRobot v1 parquet shards
# Usage:
#   python build_cost_matrix.py --in_dir lerobot_ds --out_dir matrix_v1 --alpha 1.0 --beta 2000
#
# Outputs (composite-first; primitive fallback):
#   • cost_matrix.npy  : (Q × P)  P=#unique composite plans (or H primitives if fallback)
#   • X.npy            : (Q × 5)  room context rows = [size, clutter, heat, gas, noise] (row-unit-norm)
#   • Z.npy            : (P × d)  plan (or primitive) features (row-unit-norm)
#   • meta.json

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

PRIMITIVES    = ["lidar_scan", "thermal_snap", "gas_sniff", "audio_probe"]
NOMINAL_WATTS = {"lidar_scan": 15.0, "thermal_snap": 5.0, "gas_sniff": 8.0, "audio_probe": 2.0}
SAFETY_CRIT   = {"lidar_scan": 1.0,  "thermal_snap": 0.0, "gas_sniff": 0.0, "audio_probe": 0.0}

def unit_row_norm(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n = np.maximum(n, 1e-6)
    return (M / n).astype(np.float32)

def read_context_row(df: pd.DataFrame) -> List[float]:
    # Works with your logger’s meta frame placed first in each episode
    size    = float(df["room_size"].iloc[0])    if "room_size"    in df.columns else 20.0
    clutter = float(df["room_clutter"].iloc[0]) if "room_clutter" in df.columns else 0.5
    heat    = float(df["room_heat"].iloc[0])    if "room_heat"    in df.columns else 0.0
    gas     = float(df["room_gas"].iloc[0])     if "room_gas"     in df.columns else 0.0
    noise   = float(df["room_noise"].iloc[0])   if "room_noise"   in df.columns else 0.0
    return [size, clutter, heat, gas, noise]

def main(in_dir: Path, out_dir: Path, alpha_ms_per_mWh: float, beta_ms_penalty: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) discover shards (one parquet per episode)
    files = sorted(p for p in in_dir.rglob("*.parquet"))
    if not files:
        arrow_candidates = list(in_dir.rglob("*.arrow"))
        raise SystemExit(
            f"❌ No .parquet files under {in_dir}\n"
            f"   Found {len(arrow_candidates)} .arrow files instead.\n"
            f"   Re-run your logger or pass the correct --in_dir."
        )
    print(f"📁 Found {len(files)} parquet shard(s) under {in_dir}")
    Q = len(files)

    # Quick probe for composite summaries
    have_composites = False
    for path in files[: min(5, len(files))]:
        df0 = pd.read_parquet(path, columns=None)
        if "is_summary" in df0.columns and "plan_id" in df0.columns and (df0["action"] == "composite_summary").any():
            have_composites = True
            break

    # Prepare X (room context)
    X = np.zeros((Q, 5), dtype=np.float32)

    if have_composites:
        # ─────────────────────────── COMPOSITE MODE ───────────────────────────
        # 2) Collect all unique plan_ids across all episodes
        plan_ids: List[int] = []
        for path in files:
            df = pd.read_parquet(path, columns=["action", "plan_id"])
            m  = (df["action"] == "composite_summary")
            if m.any():
                plan_ids.extend(df.loc[m, "plan_id"].astype(int).tolist())
        unique_plans = sorted(set(plan_ids))
        pid2col = {pid: j for j, pid in enumerate(unique_plans)}
        P = len(unique_plans)

        cost = np.zeros((Q, P), dtype=np.float32)
        # Feature accumulators for Z
        feat = {
            "comp_num_nodes":    np.zeros(P, dtype=np.float32),
            "comp_num_parallel": np.zeros(P, dtype=np.float32),
            "comp_depth":        np.zeros(P, dtype=np.float32),
            "mean_watts":        np.zeros(P, dtype=np.float32),
            "avg_time_ms":       np.zeros(P, dtype=np.float32),
            "includes_safety":   np.zeros(P, dtype=np.float32),
            "count":             np.zeros(P, dtype=np.float32),
        }

        for q, path in enumerate(files):
            df = pd.read_parquet(path)
            # Context
            X[q] = read_context_row(df)

            # Summaries present in this episode?
            mask_sum = (df["action"] == "composite_summary")
            if not mask_sum.any():
                continue

            df_sum = df.loc[mask_sum, ["plan_id","time_ms","energy_mWh","safety_incident",
                                       "comp_num_nodes","comp_num_parallel","comp_depth"]].copy()

            # Optional: If you later log energy in Wh instead of mWh, flip the next line to "*= 1000"
            # df_sum["energy_mWh"] = df_sum["energy_mWh"] * 1000.0  # ← enable only if needed

            # Cost per composite plan in this episode
            for _, row in df_sum.iterrows():
                pid = int(row["plan_id"])
                j   = pid2col[pid]
                t_ms = float(row["time_ms"])
                e_mWh = float(row["energy_mWh"])
                s     = bool(row["safety_incident"])

                cost[q, j] = t_ms + alpha_ms_per_mWh * e_mWh + (beta_ms_penalty if s else 0.0)

                # Accumulate Z features
                feat["comp_num_nodes"][j]     += float(row["comp_num_nodes"])
                feat["comp_num_parallel"][j]  += float(row["comp_num_parallel"])
                feat["comp_depth"][j]         += float(row["comp_depth"])
                feat["avg_time_ms"][j]        += t_ms
                feat["includes_safety"][j]    += float(s)
                feat["count"][j]              += 1.0

            # Mean watts: average nominal watts of node rows belonging to the plans seen in this episode
            if "is_summary" in df.columns and "plan_id" in df.columns and "action" in df.columns:
                df_nodes = df[df["plan_id"].isin(df_sum["plan_id"]) & (df["is_summary"] == False)]
                if not df_nodes.empty:
                    watts_series = df_nodes["action"].map(NOMINAL_WATTS).fillna(1.0).astype(float)
                    for pid, group in df_nodes.groupby("plan_id"):
                        j = pid2col[int(pid)]
                        feat["mean_watts"][j] += float(watts_series.loc[group.index].mean())

        # Finalize Z: [nodes, parallel_pairs, depth, mean_watts, avg_time_ms, includes_safety]
        cnt = np.maximum(1.0, feat["count"])
        Z = np.stack([
            feat["comp_num_nodes"]    / cnt,
            feat["comp_num_parallel"] / cnt,
            feat["comp_depth"]        / cnt,
            feat["mean_watts"]        / cnt,
            feat["avg_time_ms"]       / cnt,
            feat["includes_safety"]   / cnt,
        ], axis=1).astype(np.float32)

        # Normalize rows
        X = unit_row_norm(X)
        Z = unit_row_norm(Z)

        meta = {
            "mode": "composite",
            "num_rooms": Q,
            "num_plans": int(P),
            "Z_columns": ["comp_nodes","comp_parallel","depth","mean_watts","avg_time_ms","includes_safety"],
        }

    else:
        # ───────────────────────── PRIMITIVE FALLBACK ────────────────────────
        H = len(PRIMITIVES)
        cost = np.zeros((Q, H), dtype=np.float32)
        Z    = np.zeros((H, 4), dtype=np.float32)  # [is_lidar, watts, avg_time_ms, safety_critical]

        time_sums: Dict[str, float]   = {p: 0.0 for p in PRIMITIVES}
        time_counts: Dict[str, int]   = {p: 0   for p in PRIMITIVES}

        for q, path in enumerate(files):
            df = pd.read_parquet(path)
            X[q] = read_context_row(df)

            for h, prim in enumerate(PRIMITIVES):
                fr = df[df["action"] == prim]
                if fr.empty:
                    continue
                t_ms  = float(fr["time_ms"].iloc[0])
                e_mWh = float(fr["energy_mWh"].iloc[0])
                safe  = bool(fr["safety_incident"].iloc[0])

                time_sums[prim]   += t_ms
                time_counts[prim] += 1

                cost[q, h] = t_ms + alpha_ms_per_mWh * e_mWh + (beta_ms_penalty if safe else 0.0)

        # Build Z (primitive features) with per-primitive avg time
        avg_time_ms = []
        for h, prim in enumerate(PRIMITIVES):
            avg_t = time_sums[prim] / max(1, time_counts[prim])
            avg_time_ms.append(avg_t)
            Z[h] = [
                1.0 if prim == "lidar_scan" else 0.0,
                NOMINAL_WATTS[prim],
                avg_t,
                SAFETY_CRIT[prim],
            ]

        # Normalize rows
        X = unit_row_norm(X)
        Z = unit_row_norm(Z)

        meta = {
            "mode": "primitive",
            "num_rooms": Q,
            "num_primitives": H,
            "Z_columns": ["is_lidar","watts","avg_time_ms","safety_critical"],
            "avg_time_ms": avg_time_ms,
        }

    # 5) Save artefacts
    np.save(out_dir / "cost_matrix.npy", cost)
    np.save(out_dir / "X.npy",           X)
    np.save(out_dir / "Z.npy",           Z)
    (out_dir / "meta.json").write_text(json.dumps({
        **meta,
        "alpha_ms_per_mWh": float(alpha_ms_per_mWh),
        "beta_ms_penalty":  float(beta_ms_penalty),
    }, indent=2))
    print(f"✅ Saved → {out_dir}/cost_matrix.npy, X.npy, Z.npy, meta.json  (mode={meta['mode']})")

def build_from_dataset(*, dataset_root: str, out_dir: str, alpha: float = 1.0, beta: float = 2000.0) -> None:
    """
    Programmatic entry point used by select_and_execute.py
    Example:
        build_from_dataset(dataset_root="lerobot_exec_ds", out_dir="matrix_v3", alpha=1.0, beta=2000.0)
    """
    main(Path(dataset_root), Path(out_dir), alpha, beta)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  type=Path, default=Path("lerobot_ds"))
    ap.add_argument("--out_dir", type=Path, default=Path("matrix_v1"))
    ap.add_argument("--alpha",   type=float, default=1.0,    help="ms per mWh")
    ap.add_argument("--beta",    type=float, default=2000.0, help="ms penalty for a safety incident")
    args = ap.parse_args()
    main(args.in_dir, args.out_dir, args.alpha, args.beta)

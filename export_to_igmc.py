#!/usr/bin/env python3
# export_to_igmc.py
# Usage:
#   python export_to_igmc.py --matrix_dir matrix_v1 --out_dir igmc_data --val_frac 0.2 --bins 0

import argparse, json, numpy as np, pandas as pd
import sys
from pathlib import Path

def main(matrix_dir: Path, out_dir: Path, val_frac: float, seed: int, bins: int, class_values: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = np.load(matrix_dir / "cost_matrix.npy")              # Q × P
    X = np.load(matrix_dir / "X.npy").astype(np.float32)     # Q × d_x
    Z = np.load(matrix_dir / "Z.npy").astype(np.float32)     # P × d_z
    # quick debug: prove not-all-zero & dtype
    print(f"[debug] X dtype={X.dtype} Z dtype={Z.dtype}  "
          f"|| X row-norm min/max={np.linalg.norm(X,axis=1).min():.3f}/{np.linalg.norm(X,axis=1).max():.3f}  "
          f"Z row-norm min/max={np.linalg.norm(Z,axis=1).min():.3f}/{np.linalg.norm(Z,axis=1).max():.3f}")

    # Load meta.json (fallback to legacy primitive_meta.json if needed)
    meta_path = matrix_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        legacy = matrix_dir / "primitive_meta.json"
        if legacy.exists():
            meta = json.loads(legacy.read_text())
        else:
            meta = {}

    Q, P = L.shape

    # Only keep observed edges: treat zero as missing; if real zero-costs exist, add small epsilon or change criterion.
    obs = np.argwhere(L > 0)
    costs = L[obs[:,0], obs[:,1]].astype(np.float32)
    # ---------- DEBUG / SANITY ----------
    print(f"[debug] Found cost matrix with shape {L.shape}; observed edges count = {len(obs)}")
    # Show a few samples
    for i in range(min(5, len(obs))):
        u, v = obs[i]
        y = costs[i]
        print(f"  sample {i}: u={u} v={(v + L.shape[0])} y={y:.2f}")
    Q, P = L.shape
    assert (obs[:,0] >= 0).all() and (obs[:,0] < Q).all()
    assert (obs[:,1] >= 0).all() and (obs[:,1] < P).all()
    # -------------------------------------

    # Shuffle & split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(obs))
    n_val = int(len(idx) * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def dump_csv(which, ids):
        u = obs[ids, 0]
        v = obs[ids, 1] + Q    # IGMC convention: item ids start after users
        y = costs[ids]
        df = pd.DataFrame({"u": u, "v": v, "y": y})
        df.to_csv(out_dir / f"{which}.csv", index=False)

    dump_csv("train", tr_idx)
    dump_csv("val",   val_idx)

    # Side features
    np.save(out_dir / "A_users.npy", X)  # users (rooms/contexts)
    np.save(out_dir / "B_items.npy", Z)  # items (composite plans or primitives)

    # Relation config for IGMC:
    (out_dir / "class_values.json").write_text(json.dumps(class_values, indent=2))
    print(f"[debug] Wrote class_values.json with {class_values}")

    # Export IGMC-friendly metadata (including original mode if present)
    meta_out = dict(
        mode=meta.get("mode", "unknown"),
        num_users=Q,
        num_items=P,
        dx=int(X.shape[1]),
        dz=int(Z.shape[1]),
        n_train=int(len(tr_idx)),
        n_val=int(len(val_idx)),
    )
    (out_dir / "igmc_meta.json").write_text(json.dumps(meta_out, indent=2))

    print(f"✅ Exported IGMC data → {out_dir}  | train={meta_out['n_train']} val={meta_out['n_val']}  (users={Q}, items={P}, mode={meta_out['mode']})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix_dir", type=Path, default=Path("matrix_v2"))
    ap.add_argument("--out_dir",    type=Path, default=Path("igmc_data"))
    ap.add_argument("--val_frac",   type=float, default=0.2)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--bins",       type=int,   default=0, help="kept for compatibility; we use regression so leave 0")
    ap.add_argument("--class_values", type=int, nargs="+", default=[0],
                    help="relation class_values to write (can try [1] if [0] fails)")
    args = ap.parse_args()
    main(args.matrix_dir, args.out_dir, args.val_frac, args.seed, args.bins, args.class_values)

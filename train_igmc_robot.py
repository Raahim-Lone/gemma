#!/usr/bin/env python3
# train_igmc_robot.py
import argparse, json, numpy as np, pandas as pd, torch, scipy.sparse as sp
from pathlib import Path
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.data   import Data
from tqdm import tqdm
'''
python train_igmc_robot.py --data_root igmc_data --epochs 160 --hop 2 --batch 128 --out_dir igmc_runs
'''
# --- IGMC repo imports (make sure IGMC/ is cloned next to this script) ---
import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
from IGMC.util_functions import (
    subgraph_extraction_labeling as subg_extract,
    SparseRowIndexer, SparseColIndexer,
    construct_pyg_graph,
)
from IGMC.models import IGMC

class DummyDS:
    def __init__(self, num_features):
        self.num_features = num_features
        self.num_classes  = 1
USE_SUBGRAPH = False  # ← set True later if you really need neighborhood structure

class CostSubgraph(torch.utils.data.Dataset):
    """
    Builds a subgraph per (u, j) with single relation r=0 and continuous target y.
    """
    def __init__(self, csv_path: Path, *, hop: int, Arow, Acol, A_users, B_items, adj_csr=None, adj_csc=None):
        df = pd.read_csv(csv_path)
        self.u = df.u.to_numpy().astype(int)
        self.v = df.v.to_numpy().astype(int)
        y_raw = df.y.to_numpy().astype(np.float32)
        self.y = np.log1p(y_raw).astype(np.float32)  # train in log space
        self._y_is_log = True
        self.hop = hop
        self.Arow = Arow
        self.Acol = Acol
        self.A = A_users
        self.B = B_items
        self.n_users = A_users.shape[0]
        self.adj = adj_csr
        self.adj_T = adj_csc
        self.dummy_hits = 0

    def __len__(self): return len(self.y)

    def _dummy_graph(self, y, u=None, j=None):
        # 2 nodes: row0=user center, row1=item center; connect them
        D = 2*self.hop + 2
        x = torch.zeros((2, D), dtype=torch.float32)
        x[0, 0] = 1.0  # user center flag
        x[1, 1] = 1.0  # item center flag
        g = Data(
            x=x,
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),  # 0↔1
            edge_type=torch.zeros((2,), dtype=torch.long),
            y=torch.tensor([y], dtype=torch.float32),
        )
        # side features must be per-graph (1×d), not per-node
        # side features MUST be per-graph (1×d) so they collate to [B, d]
        if u is not None and 0 <= u < self.A.shape[0]:
            g.u_feature = torch.from_numpy(self.A[u]).to(torch.float32).unsqueeze(0)
        else:
            g.u_feature = torch.zeros((1, self.A.shape[1]), dtype=torch.float32)
        if j is not None and 0 <= j < self.B.shape[0]:
            g.v_feature = torch.from_numpy(self.B[j]).to(torch.float32).unsqueeze(0)
        else:
            g.v_feature = torch.zeros((1, self.B.shape[1]), dtype=torch.float32)
        return g

    def __getitem__(self, idx):
        u = int(self.u[idx])
        j = int(self.v[idx]) - self.n_users   # item column index in [0, P)
        r = 0  # single relation
        if not USE_SUBGRAPH:
            # Side features still flow through u_feature / v_feature
            return self._dummy_graph(self.y[idx], u=u, j=j)

        # bounds / isolation guard
        if u < 0 or u >= self.A.shape[0] or j < 0 or j >= self.B.shape[0]:
            self.dummy_hits += 1
            return self._dummy_graph(self.y[idx], u=u, j=j)
        if False and self.adj is not None and self.adj_T is not None:
            deg_u = int(self.adj.indptr[u+1] - self.adj.indptr[u])
            deg_j = int(self.adj_T.indptr[j+1] - self.adj_T.indptr[j])
            if deg_u == 0 or deg_j == 0:
                self.dummy_hits += 1
                return self._dummy_graph(self.y[idx], u=u, j=j)

        # subgraph_extraction_labeling returns:
        # (u_idx, v_idx, r, node_labels, max_node_label, y, node_features)
        extracted = None
        class_vals = np.asarray([0], dtype=np.int64)
        rel_label  = np.int64(0)
        try:
            extracted = subg_extract(
                (np.int64(u), np.int64(j)), self.Arow, self.Acol, np.int64(self.hop),
                1.0, None,
                self.A, self.B,
                class_vals,
                rel_label
            )
        except Exception:
            # Fall back to dummy (2-node) graph; side features still used.
            self.dummy_hits += 1
            return self._dummy_graph(self.y[idx], u=u, j=j)
        if extracted is None:
            self.dummy_hits += 1
            return self._dummy_graph(self.y[idx], u=u, j=j)
        u_idx, v_idx, rel, node_labels, max_label, _, node_feats = extracted
        y_tensor = torch.tensor([self.y[idx]], dtype=torch.float32)
        g = construct_pyg_graph(u_idx, v_idx, rel, node_labels, max_label, y_tensor, node_feats)
        # --- ensure feature width and center-node presence ---
        D = 2*self.hop + 2
        if g.x.dim() != 2:
            g.x = torch.zeros((0, D), dtype=torch.float32)
        if g.x.size(1) < D:
            pad = D - g.x.size(1)
            g.x = torch.cat([g.x, torch.zeros((g.x.size(0), pad), dtype=torch.float32)], dim=1)
        elif g.x.size(1) > D:
            g.x = g.x[:, :D]
        # If no nodes or no center flags, synthesize a single center row
        u_mask = (g.x[:, 0] == 1) if g.x.size(0) > 0 else torch.zeros(0, dtype=torch.bool)
        i_mask = (g.x[:, 1] == 1) if g.x.size(0) > 0 else torch.zeros(0, dtype=torch.bool)
        if g.x.size(0) < 2 or int(u_mask.sum()) != 1 or int(i_mask.sum()) != 1:
            # rebuild minimal 2-node graph with 0↔1 edge
            g.x = torch.zeros((2, D), dtype=torch.float32)
            g.x[0,0] = 1.0  # user center at row 0
            g.x[1,1] = 1.0  # item center at row 1
            g.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            g.edge_type  = torch.zeros((2,), dtype=torch.long)
        else:
            # if both centers exist but share the same row (paranoia), separate them
            if torch.nonzero(u_mask, as_tuple=False)[0].item() == torch.nonzero(i_mask, as_tuple=False)[0].item():
                x_new = torch.zeros((2, D), dtype=torch.float32)
                x_new[0,0] = 1.0; x_new[1,1] = 1.0
                g.x = x_new
                g.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                g.edge_type  = torch.zeros((2,), dtype=torch.long)
        # side features MUST be per-graph (1×d) so they collate to [B, d]
        g.u_feature = torch.from_numpy(np.ascontiguousarray(self.A[u])).to(torch.float32).unsqueeze(0)
        g.v_feature = torch.from_numpy(np.ascontiguousarray(self.B[j])).to(torch.float32).unsqueeze(0)
        return g

def build_indexers(n_users, n_items, train_csv, extra_csvs=()):
    # build adjacency from train ∪ extras (e.g., val) to reduce isolated nodes
    dfs = [pd.read_csv(train_csv)] + [pd.read_csv(p) for p in extra_csvs]
    df = pd.concat(dfs, ignore_index=True)
    # adjacency for neighborhood extraction (weights = 1)
    u = df.u.to_numpy().astype(np.int64)
    j = (df.v.to_numpy() - n_users).astype(np.int64)
    data = np.ones_like(u, dtype=np.float32)
    # ensure index arrays are int64 and contiguous
    u = np.ascontiguousarray(u, dtype=np.int64)
    j = np.ascontiguousarray(j, dtype=np.int64)
    adj = sp.csr_matrix((data, (u, j)), shape=(n_users, n_items), dtype=np.float32)
    adj_csc = adj.tocsc()
    return SparseRowIndexer(adj), SparseColIndexer(adj_csc), adj, adj_csc

def main(data_root: Path, epochs: int, hop: int, device: str, lr: float, wd: float,
         latent_dim: str, batch: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    A = np.load(data_root / "A_users.npy")   # Q × d_x
    B = np.load(data_root / "B_items.npy")   # P × d_z
    # ---- ensure proper dtype and basic normalization similar to original IGMC pipeline ----
    import numpy as _np
    from sklearn.preprocessing import StandardScaler

    def prep_side(mat):
        # Standardize (float64), then cast back to float32 and make contiguous
        mat = StandardScaler().fit_transform(_np.asarray(mat, dtype=_np.float32))
        norm = _np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / _np.maximum(1e-6, norm)
        mat = mat.astype(_np.float32, copy=False)
        return _np.ascontiguousarray(mat)

    A = prep_side(A)
    B = prep_side(B)
    print(f"[debug] side features ready: A dtype={A.dtype} B dtype={B.dtype} "
          f"A row-norm mean={np.linalg.norm(A,axis=1).mean():.3f} "
          f"B row-norm mean={np.linalg.norm(B,axis=1).mean():.3f}")

    # sanity: require nonzero side feature dims
    assert A.shape[1] > 0 and B.shape[1] > 0, "Side features have zero dimensionality!"

    Q, P = A.shape[0], B.shape[0]
    n_side = A.shape[1] + B.shape[1]

    Arow, Acol, ADJ, ADJ_T = build_indexers(Q, P, data_root / "train.csv",
                                            extra_csvs=[data_root / "val.csv"])

    # datasets & loaders
    tr_ds  = CostSubgraph(data_root / "train.csv", hop=hop, Arow=Arow, Acol=Acol,
                          A_users=A, B_items=B, adj_csr=ADJ, adj_csc=ADJ_T)
    va_ds  = CostSubgraph(data_root / "val.csv",   hop=hop, Arow=Arow, Acol=Acol,
                          A_users=A, B_items=B, adj_csr=ADJ, adj_csc=ADJ_T)
    tr_ld  = GeoLoader(tr_ds, batch_size=batch, shuffle=True)
    va_ld  = GeoLoader(va_ds, batch_size=batch)

    # model
    dims = [int(x) for x in latent_dim.split(",")]  # e.g., "32,32,32,32"
    stub = DummyDS(num_features=2*hop + 2)          # per IGMC convention
    model = IGMC(
        stub,
        latent_dim=dims,
        num_relations=1,        # single relation
        num_bases=1,
        regression=True,
        side_features=True,
        n_side_features=n_side,
        multiply_by=1
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.L1Loss()  # MAE on cost; change to MSE if you prefer

    best = {"rmse": 1e18, "epoch": 0}
    best_path = out_dir / "igmc_robot_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        print(f"[debug] dummy hits so far: train={tr_ds.dummy_hits}  val={va_ds.dummy_hits}")
        pbar = tqdm(tr_ld, desc=f"epoch {epoch:03d}", leave=False)
        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            if epoch == 1 and not hasattr(model, "_shape_printed"):
                print("u_feat mean L2:", batch.u_feature.norm(dim=1).mean().item(),
                      "v_feat mean L2:", batch.v_feature.norm(dim=1).mean().item())
                print("u_feat min/max:", float(batch.u_feature.min()), float(batch.u_feature.max()),
                      "v_feat min/max:", float(batch.v_feature.min()), float(batch.v_feature.max()))
                print("y_log stats → min/med/max:",
                      float(batch.y.min()), float(batch.y.median()), float(batch.y.max()))
                model._probe = True

                try:
                    print("x:", batch.x.shape, "u_feature:", batch.u_feature.shape, "v_feature:", batch.v_feature.shape)
                    # count center rows the model will see (first two cols are center flags)
                    users = (batch.x[:,0] == 1).sum().item()
                    items = (batch.x[:,1] == 1).sum().item()
                    print("center rows — users:", users, "items:", items)
                except Exception as e:
                    print("shape probe failed:", e)
                model._shape_printed = True

            pred  = model(batch)              # shape [B, 1]
            loss  = loss_fn(pred, batch.y)    # batch.y is [B, 1]
            optim.zero_grad(); loss.backward(); optim.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        # validation
        # validation
        model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for batch in va_ld:
                batch = batch.to(device, non_blocking=True)
                out   = model(batch)
                preds_all.append(out.cpu())
                targets_all.append(batch.y.cpu())
        if len(preds_all) == 0:
            print("⚠️  No validation batches. Check your split.")
            continue

        preds_log   = torch.cat(preds_all, 0)
        targets_log = torch.cat(targets_all, 0)

        # loss is already in log-space, but report both:
        preds_ms   = torch.expm1(preds_log)
        targets_ms = torch.expm1(targets_log)
        diff_ms    = preds_ms - targets_ms
        mae_ms     = diff_ms.abs().mean().item()
        rmse_ms    = torch.sqrt((diff_ms ** 2).mean()).item()
        mae_log    = (preds_log - targets_log).abs().mean().item()
        rmse_log   = torch.sqrt(((preds_log - targets_log) ** 2).mean()).item()

        print(f"epoch {epoch:03d}  MAE(ms)={mae_ms:.1f}  RMSE(ms)={rmse_ms:.1f}  "
              f"[log: MAE={mae_log:.3f} RMSE={rmse_log:.3f}]")

        # use rmse_ms as the selection criterion
        if rmse_ms < best["rmse"]:
            best.update({"rmse": rmse_ms, "epoch": epoch, "mae": mae_ms, "rmse_log": rmse_log})
            torch.save({
                "state_dict": model.state_dict(),
                "epoch":      epoch,
                "metrics": {
                    "mae_ms": mae_ms,
                    "rmse_ms": rmse_ms,
                    "mae_log": mae_log,
                    "rmse_log": rmse_log,
                },
                "args":       dict(epochs=epochs, hop=hop, lr=lr, wd=wd, latent_dim=latent_dim),
            }, best_path)
            print(f"✓ saved best @ epoch {epoch:03d}  RMSE(ms)={rmse_ms:.1f} → {best_path}")

    print("Done.")
    if best["epoch"] > 0:
        print(f"Best checkpoint: epoch {best['epoch']:03d}, RMSE={best['rmse']:.3f} → {best_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("igmc_data"))
    ap.add_argument("--epochs",    type=int,   default=60)
    ap.add_argument("--hop",       type=int,   default=2)
    ap.add_argument("--device",    type=str,   default=("cuda" if torch.cuda.is_available()
                                                        else "cpu"))
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--wd",        type=float, default=1e-4)
    ap.add_argument("--latent_dim",type=str,   default="32,32,32,32")
    ap.add_argument("--batch",     type=int,   default=128)
    ap.add_argument("--out_dir",   type=Path,  default=Path("igmc_runs"))
    args = ap.parse_args()
    main(**vars(args))



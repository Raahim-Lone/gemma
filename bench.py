#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-Python proxy benchmark: CQLite-like exploration vs. baselines with your fusion-MLP
- Grid world with obstacles/unknown space
- Frontier detection
- Policies: CQLite-Lite (coverage-biased), Greedy-Closest
- Optional: use your fusion model to pick a sensor composite per frontier (adds scan time)
"""

from __future__ import annotations
import argparse, math, random, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

# -----------------------------
# Grid env
# -----------------------------
FREE, OBST, UNK = 0, 1, -1

@dataclass
class SimCfg:
    size: int = 80
    obstacle_prob: float = 0.08
    n_robots: int = 2
    robot_speed: float = 1.0    # cells per second
    info_radius: int = 4
    max_steps: int = 25_000
    coverage_target: float = 0.95
    seed: int = 0

class GridSim:
    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        # Ground truth: 0 free, 1 obst
        gt = (rng.random((cfg.size, cfg.size)) < cfg.obstacle_prob).astype(np.int8)
        # carve a free “start zone”
        s = cfg.size // 10
        gt[:s,:s] = FREE
        self.gt = gt
        # what robots have “discovered”
        self.map = np.full_like(gt, UNK, dtype=np.int8)
        self._reveal((0,0), radius=3)
        # start robots in start zone corners
        self.robots = [(0,0), (s-1, s-1)][:cfg.n_robots]
        self.t = 0.0   # seconds
        self.travel_dist = [0.0 for _ in self.robots]

    def _neighbors4(self, p):
        x,y = p
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx, y+dy
            if 0 <= nx < self.cfg.size and 0 <= ny < self.cfg.size:
                yield nx,ny

    def _reveal(self, p, radius=3):
        x0,y0 = p
        for x in range(max(0,x0-radius), min(self.cfg.size, x0+radius+1)):
            for y in range(max(0,y0-radius), min(self.cfg.size, y0+radius+1)):
                self.map[x,y] = self.gt[x,y]

    def coverage(self):
        known = np.sum(self.map != UNK)
        total = self.cfg.size * self.cfg.size - np.sum(self.gt == OBST)  # free+unknown target is free space
        # we count coverage over free+unknown cells; obstacles are not expected to be traversed
        free_unknown = self.cfg.size * self.cfg.size - np.sum(self.gt==OBST)
        known_free_unknown = np.sum((self.map != UNK) & (self.gt!=OBST))
        return known_free_unknown / max(1, free_unknown)

    def frontiers(self) -> List[Tuple[int,int]]:
        # frontier = unknown cell with at least one known-free neighbor
        fr = []
        for x in range(self.cfg.size):
            for y in range(self.cfg.size):
                if self.map[x,y] == UNK:
                    for nx,ny in self._neighbors4((x,y)):
                        if self.map[nx,ny] == FREE:
                            fr.append((x,y)); break
        return fr

    def a_star_dist(self, s: Tuple[int,int], g: Tuple[int,int]) -> Optional[int]:
        # grid A*, costs=1 on FREE/UNK (assume unknown traversable until revealed)
        import heapq
        if s == g: return 0
        N = self.cfg.size
        closed = np.zeros((N,N), dtype=bool)
        pq = [(0+abs(s[0]-g[0])+abs(s[1]-g[1]), 0, s)]
        while pq:
            f,gcost,u = heapq.heappop(pq)
            if closed[u]: continue
            closed[u] = True
            if u == g: return gcost
            for v in self._neighbors4(u):
                # treat obstacles as blocked *if known obstacle*; unknown allowed (optimistic)
                if self.map[v] == OBST: continue
                if not closed[v]:
                    h = abs(v[0]-g[0])+abs(v[1]-g[1])
                    heapq.heappush(pq, (gcost+1+h, gcost+1, v))
        return None

    def step_move_and_scan(self, robot_idx: int, goal: Tuple[int,int], scan_time_s: float):
        # Move robot to goal (shortest path), reveal neighborhood along the way, accrue time by speed
        d = self.a_star_dist(self.robots[robot_idx], goal)
        if d is None: 
            return  # unreachable from current knowledge
        # “teleport” but count time and reveal a line along the path via Bresenham-like sampling
        self.travel_dist[robot_idx] += d
        self.t += d / self.cfg.robot_speed
        self._reveal(goal, radius=2)
        self.robots[robot_idx] = goal
        # scan time
        self.t += scan_time_s
        self._reveal(goal, radius=3)

# -----------------------------
# “CQLite-Lite” planner
# -----------------------------
@dataclass
class CQLiteParams:
    info_radius: int = 4
    info_multiplier: float = 3.0
    hysteresis_radius: float = 3.0
    hysteresis_gain: float = 2.0
    alpha: float = 0.2     # Q update
    gamma: float = 0.9

class CQLiteLite:
    def __init__(self, params: CQLiteParams):
        self.p = params
        self.Q: Dict[Tuple[int,int], float] = {}

    def info_gain(self, sim: GridSim, p: Tuple[int,int]) -> int:
        x0,y0 = p
        r = self.p.info_radius
        unk = 0
        for x in range(max(0,x0-r), min(sim.cfg.size, x0+r+1)):
            for y in range(max(0,y0-r), min(sim.cfg.size, y0+r+1)):
                if sim.map[x,y] == UNK: unk += 1
        return unk

    def select_goals(self, sim: GridSim, robots: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
        fr = sim.frontiers()
        if not fr: return [robots[i] for i in range(len(robots))]
        goals = []
        claimed = set()
        for ridx, rpos in enumerate(robots):
            best = None; best_val = -1e9
            for f in fr:
                if f in claimed: continue
                d = sim.a_star_dist(rpos, f)
                if d is None: continue
                info = self.info_gain(sim, f)
                hyst = 0.0
                if abs(rpos[0]-f[0]) + abs(rpos[1]-f[1]) <= self.p.hysteresis_radius:
                    hyst = self.p.hysteresis_gain
                base = self.p.info_multiplier*info - d + hyst
                q = self.Q.get(f, 0.0)
                val = base + q
                if val > best_val:
                    best_val, best = val, f
            if best is None:
                best = robots[ridx]
            goals.append(best)
            claimed.add(best)
        return goals

    def update(self, goal: Tuple[int,int], reward: float, next_best: float):
        # Q(g) <- (1-alpha) Q(g) + alpha (reward + gamma * next_best)
        q = self.Q.get(goal, 0.0)
        self.Q[goal] = (1-self.p.alpha)*q + self.p.alpha*(reward + self.p.gamma*next_best)

# -----------------------------
# Fusion-MLP hook (your model)
# -----------------------------
class FusionHook:
    """Load your fusion model and choose fastest composite; return predicted scan time (seconds)."""
    def __init__(self, fusion_dir: Path, imc_vectors: Path, good_json: Path, device: str="cpu"):
        # lazy import torch only if used
        import torch, numpy as np, json as _json
        self.torch = torch
        self.np = np
        self.imc = np.load(imc_vectors).astype(np.float32)
        self.good = _json.loads(Path(good_json).read_text())
        # load model (same logic as your runtime_loop)
        ckpt = torch.load(fusion_dir / "best_model.pt", map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        dims = self._infer_dims(state)
        self.model = self._mlp(dims); self.model.load_state_dict(state); self.model.eval()
        self.device = self.torch.device(device)
        self.model.to(self.device)
        # scalers
        npz = None
        if (fusion_dir/"scalers.npz").exists(): npz = np.load(fusion_dir/"scalers.npz")
        elif (fusion_dir/"scaler.npz").exists(): npz = np.load(fusion_dir/"scaler.npz")
        self.x_mean = self.x_std = None; self.y_mean = self.y_std = None; self.use_log = False
        if npz is not None:
            if "mu_x" in npz and "sc_x" in npz:
                self.x_mean = npz["mu_x"].astype(np.float32)
                self.x_std  = npz["sc_x"].astype(np.float32)
                # .item() ensures scalars, avoiding NumPy 1.25 deprecation warnings
                self.y_mean = (npz["mu_y"].item() if "mu_y" in npz else None)
                self.y_std  = (npz["sc_y"].item() if "sc_y" in npz else None)
                # use_log_targets may be stored as length-1 array
                self.use_log = bool(np.array(npz.get("use_log_targets", [0])).item())
            else:
                self.x_mean = (npz["x_mean"].astype(np.float32) if "x_mean" in npz.files else None)
                self.x_std  = (npz["x_std"].astype(np.float32)  if "x_std"  in npz.files else None)
                self.y_mean = (npz["y_mean"].item() if "y_mean" in npz.files else None)
                self.y_std  = (npz["y_std"].item()  if "y_std"  in npz.files else None)

    def _infer_dims(self, sd):
        pairs=[]
        for k,v in sd.items():
            if k.endswith(".weight") and v.ndim==2:
                try: idx=int(k.split(".")[1])
                except: idx=9999
                pairs.append((idx,v))
        pairs.sort(key=lambda x:x[0])
        dims=[int(pairs[0][1].shape[1])]
        for _,W in pairs: dims.append(int(W.shape[0]))
        return dims

    def _mlp(self, dims):
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self, dims):
                super().__init__()
                layers = []
                for i in range(len(dims)-2):
                    layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
                layers += [nn.Linear(dims[-2], dims[-1])]
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x)
        return M(dims)

    def predict_scan_seconds(self, ctx5: np.ndarray) -> float:
        # score all IMC, pick best pred time; convert ms->s
        N = self.imc.shape[0]
        X = np.concatenate([np.repeat(ctx5[None,:], N, axis=0), self.imc], axis=1).astype(np.float32)
        if self.x_mean is not None and self.x_std is not None:
            s = np.where(self.x_std <= 1e-8, 1.0, self.x_std).astype(np.float32)
            X = (X - self.x_mean.astype(np.float32)) / s
        xt = self.torch.from_numpy(X).to(self.device)
        with self.torch.no_grad():
            yh = self.model(xt).squeeze(-1).cpu().numpy()
        if (self.y_mean is not None) and (self.y_std is not None):
            yh = yh * np.float32(self.y_std) + np.float32(self.y_mean)
        if self.use_log:
            yh = np.expm1(yh)
        best_ms = float(np.min(yh))
        return max(0.0, best_ms / 1000.0)
    def choose_composite(self,
                         ctx5: np.ndarray,
                         *,
                         llm_model: Optional[str] = None,
                         top_k: int = 0) -> Tuple[int, float]:
        """
        Return: (chosen_index, scan_seconds)
        """
        # compute all predictions once
        N = self.imc.shape[0]
        X = np.concatenate([np.repeat(ctx5[None,:], N, axis=0), self.imc], axis=1).astype(np.float32)
        if self.x_mean is not None and self.x_std is not None:
            s = np.where(self.x_std <= 1e-8, 1.0, self.x_std).astype(np.float32)
            X = (X - self.x_mean.astype(np.float32)) / s
        xt = self.torch.from_numpy(X).to(self.device)
        with self.torch.no_grad():
            yh = self.model(xt).squeeze(-1).cpu().numpy()
        if (self.y_mean is not None) and (self.y_std is not None):
            yh = yh * np.float32(self.y_std) + np.float32(self.y_mean)
        if self.use_log:
            yh = np.expm1(yh)
        order = np.argsort(yh)  # ms ascending
        if llm_model and top_k > 0:
            K = int(min(top_k, len(order)))
            pool = [int(i) for i in order[:K]]
            pool_cands = [
                {"index": int(i), "pred_ms": float(yh[i]), "steps": self.good[i]["parsed_plan"]}
                for i in pool
            ]
            # small, local version (duplicate to avoid import cycle)
            def _llm_rerank_local(ctx, cands, model):
                import subprocess, json as _json
                JSON_ONLY = "Return ONLY a JSON list (e.g., [3,1,2]). No prose."
                lines = [
                    "Choose best sensor-composite for fast, safe coverage.",
                    f"Context: size_m2={ctx[0]:.1f}, clutter={ctx[1]:.2f}, heat={int(ctx[2])}, gas={int(ctx[3])}, noise={int(ctx[4])}.",
                    "Prefer low time unless hazards justify thermal/gas first; consider diminishing returns & parallel steps.",
                    JSON_ONLY, "", "Candidates:"]
                for c in cands:
                    flat = [p for step in c["steps"] for p in step][:8]
                    lines.append(f'- id={c["index"]}, predicted_ms={int(c["pred_ms"])}, steps={flat}')
                prompt = "\n".join(lines)
                try:
                    proc = subprocess.run(["ollama","run",model], input=prompt, text=True, capture_output=True, timeout=20.0)
                    txt = proc.stdout.strip()
                    if txt.startswith("```"):
                        txt = txt.strip("`").strip()
                        if "\n" in txt: txt = txt.split("\n",1)[1].strip()
                    out = _json.loads(txt)
                    if isinstance(out, list) and all(isinstance(i,int) for i in out):
                        keep = {c["index"] for c in cands}
                        ordered = [i for i in out if i in keep]
                        if ordered: return ordered
                except Exception:
                    return None
                return None
            new = _llm_rerank_local(ctx5, pool_cands, llm_model)
            if new:
                idx = int(new[0])
                return idx, max(0.0, float(yh[idx])/1000.0)
        # fallback: pure fusion
        idx = int(order[0])
        return idx, max(0.0, float(yh[idx])/1000.0)

# -----------------------------
# Runners
# -----------------------------
def run_episode(policy_name: str,
                sim: GridSim,
                fusion: Optional[FusionHook],
                *,
                ig_lambda: float = 1.0) -> Dict[str,float]:
    if policy_name == "cqlite":
        pol = CQLiteLite(CQLiteParams(info_radius=sim.cfg.info_radius))
    known_prev = sim.coverage()
    for step in range(sim.cfg.max_steps):
        cov = sim.coverage()
        if cov >= sim.cfg.coverage_target: break
        # goals
        if policy_name == "cqlite":
            goals = pol.select_goals(sim, sim.robots)
        elif policy_name in ("closest", "nf"):
            fr = sim.frontiers()
            goals=[]
            for ridx, rpos in enumerate(sim.robots):
                if not fr: goals.append(rpos); continue
                best=min(fr, key=lambda f: (sim.a_star_dist(rpos,f) or 1e9))
                goals.append(best)
        elif policy_name in ("ig_ratio", "ig_minus"):
            # Yamauchi-style frontiers; Burgard-style utility
            # IG = count of unknown cells in radius (same notion used by CQLiteLite.info_gain)
            fr = sim.frontiers()
            goals=[]
            claimed=set()
            for ridx, rpos in enumerate(sim.robots):
                if not fr: goals.append(rpos); continue
                best=None; best_val=-1e18
                for f in fr:
                    if f in claimed: continue
                    d = sim.a_star_dist(rpos, f)
                    if d is None: continue
                    # information gain: number of unknown cells in a radius
                    x0,y0=f; r=sim.cfg.info_radius
                    unk=0
                    for x in range(max(0,x0-r), min(sim.cfg.size, x0+r+1)):
                        for y in range(max(0,y0-r), min(sim.cfg.size, y0+r+1)):
                            if sim.map[x,y] == UNK: unk += 1
                    if policy_name == "ig_ratio":
                        val = unk / (d + 1.0)
                    else:  # "ig_minus"
                        val = float(unk) - ig_lambda * float(d)
                    if val > best_val:
                        best_val, best = val, f
                goals.append(best if best is not None else rpos)
                if best is not None:
                    claimed.add(best)

        else:
            raise ValueError("unknown policy")

        # execute: for each robot, compute scan time (fusion if available else fixed)
        for ridx, g in enumerate(goals):
            # simple synthetic ctx5 from local map stats to feed your model
            # [size_m2, clutter, heat, gas, noise] — we fake these deterministically
            size_m2 = float(sim.cfg.size)  # proxy
            clutter = float(np.mean(sim.gt == OBST))
            heat = 0.0; gas=0.0; noise=0.0  # off by default
            ctx5 = np.array([size_m2, clutter, heat, gas, noise], dtype=np.float32)

            scan_s = 1.0
            if fusion is not None:
                use_llm = (getattr(run_episode, "_llm_every", 1) > 0 and
                           (step % getattr(run_episode, "_llm_every", 1) == 0) and
                           getattr(run_episode, "_llm_model", None) and
                           getattr(run_episode, "_llm_topk", 0) > 0)
                if use_llm:
                    _, scan_s = fusion.choose_composite(ctx5,
                                                        llm_model=getattr(run_episode, "_llm_model"),
                                                        top_k=getattr(run_episode, "_llm_topk"))
                else:
                    scan_s = fusion.predict_scan_seconds(ctx5)
            sim.step_move_and_scan(ridx, g, scan_s)

        # (Optional) Q update per chosen goal
        if policy_name == "cqlite":
            cov_now = sim.coverage()
            reward = (cov_now - known_prev) * 1000.0  # reward = uncovered area revealed
            known_prev = cov_now
            # bootstrap with current best Q
            nb = max(pol.Q.values()) if pol.Q else 0.0
            for g in set(goals):
                pol.update(g, reward, nb)

    return dict(
        steps=step+1,
        time_s=sim.t,
        coverage=sim.coverage(),
        dist=sum(sim.travel_dist),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--robots", type=int, default=2)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--fusion_dir", type=Path, default=None)
    ap.add_argument("--imc_vectors", type=Path, default=None)
    ap.add_argument("--good_json", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--policies", type=str,
                    default="closest,cqlite,ig_ratio,ig_minus",
                    help="Comma-separated: closest|nf|cqlite|ig_ratio|ig_minus")
    ap.add_argument("--ig_lambda", type=float, default=1.0,
                    help="λ in IG − λ·distance (only for ig_minus)")
    ap.add_argument("--llm_model", type=str, default=None, help="e.g., gemma3n:e4b (uses `ollama run`).")
    ap.add_argument("--llm_topk", type=int, default=0, help="If >0, re-rank fusion top-K each decision.")
    ap.add_argument("--llm_every", type=int, default=1, help="Re-rank every N decisions (reduce prompting).")

    args = ap.parse_args()

    fusion = None
    if args.fusion_dir and args.imc_vectors and args.good_json:
        fusion = FusionHook(args.fusion_dir, args.imc_vectors, args.good_json, device=args.device)

    rng = random.Random(0)
    out = []
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for ep in range(args.episodes):
        for policy in policies:
            cfg = SimCfg(size=args.size, n_robots=args.robots, coverage_target=args.target, seed=ep)
            sim = GridSim(cfg)
            setattr(run_episode, "_llm_model", args.llm_model)
            setattr(run_episode, "_llm_topk", max(0, int(args.llm_topk)))
            setattr(run_episode, "_llm_every", max(1, int(args.llm_every)) if args.llm_topk > 0 and args.llm_model else 0)

            metrics = run_episode(policy, sim, fusion=fusion, ig_lambda=args.ig_lambda)
            metrics["policy"] = policy
            metrics["ep"] = ep
            out.append(metrics)
            print(f"[ep {ep:02d}] {policy:8s}  time={metrics['time_s']:.1f}s  "
                  f"dist={metrics['dist']:.1f}  coverage={metrics['coverage']*100:.1f}%")

    # Aggregate
    import statistics as st
    def agg(key, policy):
        vals=[r[key] for r in out if r["policy"]==policy]
        return dict(mean=st.mean(vals), std=(st.pstdev(vals) if len(vals)>1 else 0.0))
    for pol in policies:
        a=agg("time_s",pol); b=agg("dist",pol)
        print(f"\n{pol}: time {a['mean']:.1f}±{a['std']:.1f}s, "
              f"distance {b['mean']:.1f}±{b['std']:.1f} cells")

if __name__ == "__main__":
    main()

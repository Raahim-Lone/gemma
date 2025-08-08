#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runtime_loop.py

Subcommands
-----------
score:
  Score all IMC candidates for a single provided context (or a sampled RoomProfile)
  using the trained fusion MLP. Writes ranked predictions.

deploy:
  For each of R rooms, create a RoomProfile (context), score all candidates,
  pick top-K (optionally diversity via k-center), execute the plans into a
  LeRobotDataset, and optionally rebuild the cost matrix by importing your
  builder (no shell-out).

Examples
--------
# 1) Score candidates for a custom context (size, clutter, heat, gas, noise)
python runtime_loop.py score \
  --fusion_dir fusion_runs \
  --imc_vectors imc_out/composites.npy \
  --good_json imc_out/good_rewrites.json \
  --ctx 22.0,0.35,0,1,0 \
  --top 10 \
  --out scored.json

# 2) Deploy: 40 rooms, pick top-3 per room with diversity, write dataset + matrices
python runtime_loop.py deploy \
  --fusion_dir fusion_runs \
  --imc_vectors imc_out/composites.npy \
  --good_json imc_out/good_rewrites.json \
  --rooms 40 --per_room 3 --diversity \
  --dataset_out runtime_ds --maybe_wipe \
  --matrix_builder_module build_cost_matrix \
  --matrix_builder_fn build_from_dataset \
  --matrix_out matrix_runtime
"""
from __future__ import annotations
import argparse, json, math, os, random, importlib, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import subprocess, textwrap, json as _json
import time as _time
import hashlib
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Import your simulation helpers if available
# -----------------------------------------------------------------------------
RunSkillFn = None
FEATURES: Dict[str, Dict[str, Any]] = {}
RoomProfile = None
BatterySim = None
JSON_ONLY_INSTR = "Return ONLY a JSON list (e.g., [3,1,2]). No prose."
_LLM_CACHE: Dict[str, List[int]] = {}
_EMA_LLM_MS: float = 2000.0      # init guess; updated after real calls
_EMA_ALPHA:  float = 0.15        # smoothing for latency EMA
# Abbreviations to shrink prompt token count
_STEP_ABBR = {
    "lidar_scan": "L",
    "thermal_snap": "T",
    "gas_sniff": "G",
    "audio_probe": "A",
    "wait": "W",
}


def _step_sig(steps: List[List[str]], max_tokens: int = 8) -> str:
    # Flatten and abbreviate primitives, cap length
    flat = [ _STEP_ABBR.get(p, p[:1].upper()) for step in steps for p in step ]
    return "".join(flat[:max_tokens])


def _ctx_sig(ctx: np.ndarray) -> str:
    # Compact numeric context string
    return f"{ctx[0]:.0f},{ctx[1]:.2f},{int(ctx[2])},{int(ctx[3])},{int(ctx[4])}"


def _should_call_llm(*, gap_pct: float, hazard: bool,
                     top1_ms: float, top2_ms: float,
                     allow_when_hazard: bool,
                     ema_llm_ms: float,
                     gain_vs_cost_ratio: float,
                     hazard_gap_relax: float) -> bool:
    """
    Call the LLM only if expected benefit likely exceeds cost.
    - Always consider calling when fusion is unsure (small gap).
    - When hazard, allow a bit more often but still require benefit.
    """
    # If fusion is very confident (big gap), usually skip
    if gap_pct >= 0.10 and not (allow_when_hazard and hazard):
        return False
    # Rough benefit proxy: how much faster #1 is vs #2 (if LLM might swap)
    expected_gain = max(0.0, top2_ms - top1_ms)
    # Hazard can relax the required margin a bit
    relax = hazard_gap_relax if hazard and allow_when_hazard else 0.0
    # Require gain to amortize LLM latency
    need = gain_vs_cost_ratio * max(500.0, ema_llm_ms)  # floor at 500 ms
    return (gap_pct < (0.10 + relax)) and (expected_gain > need)

def _round_ctx_for_key(ctx: np.ndarray) -> Tuple:
    """Quantize context for caching: size→nearest 5m², clutter→0.1 bins, hazards/noise as ints."""
    size_q = int(round(float(ctx[0]) / 5.0) * 5)
    clut_q = round(float(ctx[1]) / 0.1, 0) * 0.1
    heat_q = int(round(float(ctx[2])))
    gas_q  = int(round(float(ctx[3])))
    noise_q= int(round(float(ctx[4])))
    return (size_q, clut_q, heat_q, gas_q, noise_q)

def _llm_cache_key(ctx: np.ndarray, pool_indices: List[int]) -> str:
    key_tuple = (_round_ctx_for_key(ctx), tuple(int(i) for i in pool_indices))
    raw = _json.dumps(key_tuple, separators=(",", ":"), sort_keys=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _try_import_sim():
    global RunSkillFn, FEATURES, RoomProfile, BatterySim
    try:
        sim = importlib.import_module("log_sim_le_robot")
        RunSkillFn = sim.run_skill
        FEATURES = sim.FEATURES
        RoomProfile = sim.RoomProfile
        BatterySim = sim.BatterySim
        print("[runtime] Using run_skill/FEATURES from log_sim_le_robot.py")
        return True
    except Exception as e:
        print(f"[runtime][warn] Could not import log_sim_le_robot.py ({e}). Using minimal fallbacks.")
        return False

_loaded = _try_import_sim()
if not _loaded:
    # Minimal safe fallbacks (match your previous shapes/semantics)
    import time as _t

    class RoomProfile:
        def __init__(self, idx: int):
            self.room_id  = f"room_{idx:04d}"
            self.size_m2  = random.uniform(10, 40)
            self.clutter  = random.uniform(0.0, 1.0)
            self.heat     = random.random() < 0.3
            self.gas      = random.random() < 0.25
            self.noise    = random.random() < 0.4

    class BatterySim:
        def __init__(self, capacity_wh: float = 50.0):
            self.capacity_wh = capacity_wh
            self.remaining_wh = capacity_wh
        def drain(self, watts: float, dt: float):
            self.remaining_wh = max(self.remaining_wh - watts * dt / 3600.0, 0)
        def read(self) -> float:
            return self.remaining_wh

    def _sleep(dt: float): _t.sleep(max(0.0, dt))
    def _lidar(room, dt): 
        pts = np.random.normal(2.0, 0.3, 800)
        pts[np.random.rand(800) < room.clutter*0.4] *= 0.25
        _sleep(dt); return pts
    def _therm(room, dt):
        frm = np.random.normal(23, 1, (8,8))
        if room.heat: frm.flat[np.random.randint(0,64)] = 48 + np.random.rand()*5
        _sleep(dt); return frm
    def _gas(room, dt): _sleep(dt); return np.random.normal(200,20) + (350 if room.gas else 0)
    def _audio(room, dt): _sleep(dt); return np.random.normal(0.03,0.01) + (0.08 if room.noise else 0)

    def _run_skill(name: str, room: RoomProfile, battery: BatterySim, *, parallel_scale: float = 1.0) -> Dict[str, Any]:
        start = _t.time()
        batt0 = battery.read()
        if name == "wait":
            _sleep(0.6*parallel_scale); hazard=0; safety=0; watts=1; dt=0.6*parallel_scale
        elif name == "lidar_scan":
            pts=_lidar(room, 2.0*parallel_scale); hazard=int(pts.min()<0.3); safety=int(random.random()<room.clutter*0.05); watts=15; dt=2.0*parallel_scale
        elif name == "thermal_snap":
            frm=_therm(room, 0.8*parallel_scale); hazard=int(frm.max()>45); safety=0; watts=5; dt=0.8*parallel_scale
        elif name == "gas_sniff":
            ppm=_gas(room, 1.0*parallel_scale); hazard=int(ppm>400); safety=0; watts=8; dt=1.0*parallel_scale
        else:
            rms=_audio(room, 0.5*parallel_scale); hazard=int(rms>0.08); safety=0; watts=2; dt=0.5*parallel_scale
        battery.drain(watts, dt)
        return dict(
            observation     = np.zeros((1,), dtype=np.float32),
            action          = name,
            reward          = np.array([-dt], dtype=np.float32),
            time_ms         = np.array([int(dt*1000)], dtype=np.int64),
            energy_mWh      = np.array([batt0 - battery.read()], dtype=np.float32),
            hazard_found    = np.array([bool(hazard)]),
            safety_incident = np.array([bool(safety)]),
            room_size       = np.array([room.size_m2], dtype=np.float32),
            room_clutter    = np.array([room.clutter], dtype=np.float32),
            room_heat       = np.array([1.0 if room.heat else 0.0], dtype=np.float32),
            room_gas        = np.array([1.0 if room.gas else 0.0], dtype=np.float32),
            room_noise      = np.array([1.0 if room.noise else 0.0], dtype=np.float32),
        )
    RunSkillFn = _run_skill

    FEATURES = {
        "observation":     {"dtype": "float32", "shape": (1,), "names": ["val"]},
        "action":          {"dtype": "string",  "shape": (1,), "names": None},
        "reward":          {"dtype": "float32", "shape": (1,), "names": None},
        "time_ms":         {"dtype": "int64",   "shape": (1,), "names": None},
        "energy_mWh":      {"dtype": "float32", "shape": (1,), "names": None},
        "hazard_found":    {"dtype": "bool",    "shape": (1,), "names": None},
        "safety_incident": {"dtype": "bool",    "shape": (1,), "names": None},
        "room_size":       {"dtype": "float32", "shape": (1,), "names": None},
        "room_clutter":    {"dtype": "float32", "shape": (1,), "names": None},
        "room_heat":       {"dtype": "float32", "shape": (1,), "names": None},
        "room_gas":        {"dtype": "float32", "shape": (1,), "names": None},
        "room_noise":      {"dtype": "float32", "shape": (1,), "names": None},
        "plan_id":         {"dtype": "int64",   "shape": (1,), "names": None},
        "plan_node":       {"dtype": "int64",   "shape": (1,), "names": None},
        "parallel_group":  {"dtype": "int64",   "shape": (1,), "names": None},
        "is_summary":      {"dtype": "bool",    "shape": (1,), "names": None},
        "comp_num_nodes":  {"dtype": "int64",   "shape": (1,), "names": None},
        "comp_num_parallel":{"dtype": "int64",  "shape": (1,), "names": None},
        "comp_depth":      {"dtype": "int64",   "shape": (1,), "names": None},
    }

# -----------------------------------------------------------------------------
# Primitive aliasing (match rewrite_pipeline + sim)
# -----------------------------------------------------------------------------
PRIMITIVE_ALIAS = {"movement_pause":"wait","pause":"wait","idle":"wait","noop":"wait"}
ALLOWED_PRIMS = {"lidar_scan","thermal_snap","gas_sniff","audio_probe","wait"}

def _normalize_primitive(p: str) -> str:
    p = PRIMITIVE_ALIAS.get(p.strip(), p.strip())
    if p not in ALLOWED_PRIMS:
        raise ValueError(f"Unknown primitive: {p}")
    return p

# -----------------------------------------------------------------------------
# Fusion MLP loader (reconstruct architecture from checkpoint if needed)
# -----------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def _infer_dims_from_state_dict(sd: Dict[str, torch.Tensor]) -> List[int]:
    # Find linear layers in order: net.0.weight, net.2.weight, ...
    pairs = []
    for k,v in sd.items():
        if k.endswith(".weight") and v.ndim == 2:
            # extract index like 'net.0.weight' -> 0
            try:
                idx = int(k.split(".")[1])
            except Exception:
                idx = 9999
            pairs.append((idx, v))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        raise RuntimeError("Cannot infer MLP dims from state_dict")
    dims = [int(pairs[0][1].shape[1])]
    for _,W in pairs:
        dims.append(int(W.shape[0]))
    return dims  # [in, h1, ..., out]

def _device_from_arg(arg: str) -> str:
    if arg == "auto":
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return arg

def load_fusion(fusion_dir: Path, device: str) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt_path = fusion_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    # state dict may be saved directly or nested under 'model'/'state_dict'
    state = sd.get("state_dict", sd.get("model_state_dict", sd))
    dims: Optional[List[int]] = None

    # Optional config file written by trainer (if present)
    cfg = {}
    cfg_path = fusion_dir / "model_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            if "dims" in cfg and isinstance(cfg["dims"], list):
                dims = [int(x) for x in cfg["dims"]]
        except Exception:
            pass

    if dims is None:
        dims = _infer_dims_from_state_dict(state)

    model = SimpleMLP(dims)
    model.load_state_dict(state)
    model.to(device).eval()

    # Load scaler if available
    scaler = {"x_mean": None, "x_std": None, "y_mean": None, "y_std": None, "use_log_targets": False}
    s_npz_primary = fusion_dir / "scalers.npz"    # current
    s_npz_fallback = fusion_dir / "scaler.npz"     # in case of variant
    s_json = fusion_dir / "scaler.json"
    if s_npz_primary.exists() or s_npz_fallback.exists():
        path = s_npz_primary if s_npz_primary.exists() else s_npz_fallback
        try:
            _npz = np.load(path)
            if "mu_x" in _npz and "sc_x" in _npz:
                # trainer stored in its own naming
                scaler["x_mean"] = _npz["mu_x"].tolist()
                scaler["x_std"]  = _npz["sc_x"].tolist()
                scaler["y_mean"] = _npz["mu_y"].item() if "mu_y" in _npz else None
                scaler["y_std"]  = _npz["sc_y"].item() if "sc_y" in _npz else None
                # stored as array([0]) or array([1])
                scaler["use_log_targets"] = bool(_npz.get("use_log_targets", [0]).item())
            else:
                # fallback to expected keys
                scaler["x_mean"] = _npz["x_mean"].tolist() if "x_mean" in _npz else None
                scaler["x_std"]  = _npz["x_std"].tolist()  if "x_std"  in _npz else None
                scaler["y_mean"] = float(_npz["y_mean"])   if "y_mean" in _npz else None
                scaler["y_std"]  = float(_npz["y_std"])    if "y_std"  in _npz else None
            print(f"[runtime] Loaded scaler from {path.name}")
        except Exception as e:
            print(f"[runtime][warn] Failed to read {path.name}: {e}")
    elif s_json.exists():
        try:
            loaded = json.loads(s_json.read_text())
            scaler.update(loaded)  # assume it already uses x_mean/x_std etc.
            print(f"[runtime] Loaded scaler from {s_json.name}")
        except Exception as e:
            print(f"[runtime][warn] Failed to read {s_json.name}: {e}")
    else:
        print("[runtime][warn] No scaler file found (scalers.npz|scaler.npz|scaler.json); "
              "predictions will be in model’s standardized units.")
    return model, scaler

def _standardize(x: np.ndarray, mean: Optional[List[float]], std: Optional[List[float]]) -> np.ndarray:
    if mean is None or std is None: return x
    m = np.asarray(mean, dtype=np.float32)
    s = np.asarray(std, dtype=np.float32)
    s = np.where(s <= 1e-8, 1.0, s)
    return (x - m) / s



def _destandardize(y: np.ndarray, mean: Optional[float], std: Optional[float]) -> np.ndarray:
    if mean is None or std is None:
        return y
    return y.astype(np.float32) * np.float32(std) + np.float32(mean)

# -----------------------------------------------------------------------------
# K-center for diversity (optional)
# -----------------------------------------------------------------------------
def k_center_greedy(M: np.ndarray, k: int, *, seed: int = 0, init_idx: Optional[int] = None) -> List[int]:
    n = M.shape[0]
    if k <= 0: return []
    if k >= n: return list(range(n))
    rng = random.Random(seed)
    if init_idx is None:
        init_idx = rng.randrange(n)
    sel = [init_idx]
    dmin = np.linalg.norm(M - M[init_idx:init_idx+1], axis=1)
    while len(sel) < k:
        cand = int(np.argmax(dmin))
        sel.append(cand)
        dnew = np.linalg.norm(M - M[cand:cand+1], axis=1)
        dmin = np.minimum(dmin, dnew)
    return sel

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _load_good(good_json: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(good_json).read_text())
    if not isinstance(data, list):
        raise ValueError("good_rewrites.json must contain a list")
    return data

def _compose_ctx_from_room(room: RoomProfile) -> np.ndarray:
    return np.array([
        float(room.size_m2),
        float(room.clutter),
        1.0 if room.heat else 0.0,
        1.0 if room.gas else 0.0,
        1.0 if room.noise else 0.0,
    ], dtype=np.float32)

def _compose_feature(room_ctx: np.ndarray, imc_vec: np.ndarray) -> np.ndarray:
    return np.concatenate([room_ctx.astype(np.float32), imc_vec.astype(np.float32)], axis=0)

# -----------------------------------------------------------------------------
# Plan conversion + execution (mirrors select_and_execute)
# -----------------------------------------------------------------------------
def _steps_from_parsed_plan(parsed_plan: List[List[str]]) -> List[List[str]]:
    return [[_normalize_primitive(p) for p in step] for step in parsed_plan]

def _maybe_init_dataset(root: Path, maybe_wipe: bool) -> "LeRobotDataset":
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if root.exists() and maybe_wipe:
        import shutil; shutil.rmtree(root)
    if root.exists():
        try:
            ds = LeRobotDataset.load(root=str(root))
            print(f"[runtime] Loaded existing LeRobotDataset at {root}")
            return ds
        except Exception:
            import shutil; shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id="runtime_rooms", root=str(root), fps=10, features=FEATURES)
    print(f"[runtime] Created new LeRobotDataset at {root}")
    return ds

def _append_episode(ds: "LeRobotDataset", room_idx: int, plans: List[List[List[str]]]) -> Tuple[int,int]:
    room = RoomProfile(room_idx)
    battery = BatterySim()

    ds.clear_episode_buffer()
    ds.add_frame(
        dict(
            observation     = np.zeros((1,), dtype=np.float32),
            action          = "metadata",
            reward          = np.array([0.0], dtype=np.float32),
            time_ms         = np.array([0], dtype=np.int64),
            energy_mWh      = np.array([0.0], dtype=np.float32),
            hazard_found    = np.array([False]),
            safety_incident = np.array([False]),
            room_size       = np.array([room.size_m2], dtype=np.float32),
            room_clutter    = np.array([room.clutter], dtype=np.float32),
            room_heat       = np.array([1.0 if room.heat  else 0.0], dtype=np.float32),
            room_gas        = np.array([1.0 if room.gas   else 0.0], dtype=np.float32),
            room_noise      = np.array([1.0 if room.noise else 0.0], dtype=np.float32),
            plan_id         = np.array([-1], dtype=np.int64),
            plan_node       = np.array([-1], dtype=np.int64),
            parallel_group  = np.array([-1], dtype=np.int64),
            is_summary      = np.array([False]),
            comp_num_nodes  = np.array([0], dtype=np.int64),
            comp_num_parallel=np.array([0], dtype=np.int64),
            comp_depth      = np.array([1], dtype=np.int64),
        ),
        task="simulation",
    )

    plan_counter = 0
    total_episode_ms = 0.0

    for parsed in plans:
        flat = [p for step in parsed for p in step]
        num_nodes = len(flat)
        num_parallel = sum(1 for step in parsed if len(step) > 1)
        depth = 2 if num_parallel > 0 else 1

        pid = plan_counter; plan_counter += 1
        comp_ms, comp_mWh, any_safe, any_haz = 0.0, 0.0, False, False

        gid_counter = -1
        node_counter = 0
        for step in parsed:
            is_par = len(step) > 1
            gid = -1
            if is_par:
                gid_counter += 1; gid = gid_counter
            scale = 0.7 if is_par else 1.0
            for prim in step:
                fr = RunSkillFn(prim, room, battery, parallel_scale=scale)
                ds.add_frame(
                    dict(
                        **fr,
                        plan_id=np.array([pid], dtype=np.int64),
                        plan_node=np.array([node_counter], dtype=np.int64),
                        parallel_group=np.array([gid], dtype=np.int64),
                        is_summary=np.array([False]),
                        comp_num_nodes=np.array([num_nodes], dtype=np.int64),
                        comp_num_parallel=np.array([num_parallel], dtype=np.int64),
                        comp_depth=np.array([depth], dtype=np.int64),
                    ),
                    task="simulation",
                )
                comp_ms  += float(fr["time_ms"][0])
                comp_mWh += float(fr["energy_mWh"][0])
                any_safe |= bool(fr["safety_incident"][0])
                any_haz  |= bool(fr["hazard_found"][0])
                node_counter += 1

        summary = dict(
            observation     = np.zeros((1,), dtype=np.float32),
            action          = "composite_summary",
            reward          = np.array([-(comp_ms/1000.0)], dtype=np.float32),
            time_ms         = np.array([int(comp_ms)], dtype=np.int64),
            energy_mWh      = np.array([comp_mWh], dtype=np.float32),
            hazard_found    = np.array([any_haz]),
            safety_incident = np.array([any_safe]),
            room_size       = np.array([room.size_m2], dtype=np.float32),
            room_clutter    = np.array([room.clutter], dtype=np.float32),
            room_heat       = np.array([1.0 if room.heat else 0.0], dtype=np.float32),
            room_gas        = np.array([1.0 if room.gas else 0.0], dtype=np.float32),
            room_noise      = np.array([1.0 if room.noise else 0.0], dtype=np.float32),
            plan_id         = np.array([pid], dtype=np.int64),
            plan_node       = np.array([-1], dtype=np.int64),
            parallel_group  = np.array([-1], dtype=np.int64),
            is_summary      = np.array([True]),
            comp_num_nodes  = np.array([num_nodes], dtype=np.int64),
            comp_num_parallel=np.array([num_parallel], dtype=np.int64),
            comp_depth      = np.array([depth], dtype=np.int64),
        )
        ds.add_frame(summary, task="simulation")
        total_episode_ms += comp_ms

    ds.save_episode()
    return int(total_episode_ms), plan_counter

# -----------------------------------------------------------------------------
# SCORING
# -----------------------------------------------------------------------------
def score_all_for_context(
    model: nn.Module,
    scaler: Dict[str, Any],
    ctx: np.ndarray,
    imc: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    ctx: (5,)
    imc: (N,6)
    Returns: y_pred_ms (N,)
    """
    N = imc.shape[0]
    X = np.concatenate([np.repeat(ctx[None,:], N, axis=0), imc], axis=1).astype(np.float32)
    Xs = _standardize(X, scaler.get("x_mean"), scaler.get("x_std")).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(Xs).to(device)
        yh = model(xt).squeeze(-1).cpu().numpy()
    y_ms = _destandardize(yh, scaler.get("y_mean"), scaler.get("y_std"))
    if scaler.get("use_log_targets", False):
        # training used log1p(target); undo it
        y_ms = np.expm1(y_ms)
    return y_ms.astype(np.float32)

def _llm_rerank(ctx: np.ndarray,
                candidates: List[Dict[str, Any]],
                *,
                model: Optional[str] = None,
                temperature: float = 0.0,
                timeout: float = 25.0) -> Tuple[Optional[List[int]], float]:
    """
    candidates: list of {index:int, pred_ms:float, steps: List[List[str]]}
    Return: new order of indices (absolute indices into IMC/good arrays), or None on failure.
    """
    if not model:
        return None, 0.0
    # Compact, latency-friendly prompt
    ctxs = _ctx_sig(ctx)
    lines = [
        f"CTX:{ctxs}",
        "GOAL: pick best tradeoff of speed vs hazard. Prefer lower pred_ms unless heat/gas imply sensing first. Avoid redundant waits.",
        "Return ONLY JSON list of candidate ids in preferred order, e.g., [3,1,2].",
        "CANDIDATES:"
    ]
    for c in candidates:
        sig = _step_sig(c["steps"], max_tokens=8)
        lines.append(f'{c["index"]}:{int(c["pred_ms"])},{sig}')
    prompt = "\n".join(lines)
    try:
        t0 = _time.time()
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        llm_ms = ( _time.time() - t0 ) * 1000.0
        txt = proc.stdout.strip()
        # Some models wrap code fences; strip them.
        if txt.startswith("```"):
            txt = txt.strip("`").strip()
            # drop possible language label
            if "\n" in txt:
                txt = txt.split("\n", 1)[1].strip()
        # Expect a JSON list
        out = _json.loads(txt)
        if isinstance(out, list) and all(isinstance(i, int) for i in out):
            keep = {c["index"] for c in candidates}
            ordered = [i for i in out if i in keep]
            # pad with any missing ids in original order
            missing = [c["index"] for c in candidates if c["index"] not in ordered]
            return ordered + missing, float(llm_ms)
    except Exception as e:
        print(f"[llm][warn] rerank failed: {e}")
    return None, 0.0

# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
def cmd_score(args: argparse.Namespace) -> None:
    global _EMA_LLM_MS
    device = _device_from_arg(args.device)
    model, scaler = load_fusion(Path(args.fusion_dir), device)

    imc = np.load(args.imc_vectors).astype(np.float32)
    good = _load_good(args.good_json)

    if args.ctx:
        vals = [float(x) for x in args.ctx.split(",")]
        if len(vals) != 5: raise ValueError("--ctx must have 5 numbers: size,clutter,heat,gas,noise")
        ctx = np.array(vals, dtype=np.float32)
    else:
        rp = RoomProfile(0)
        ctx = _compose_ctx_from_room(rp)

    y_ms = score_all_for_context(model, scaler, ctx, imc, device)
    print(f"[score] ctx={ctx.tolist()}  y_ms[min/median/max]={y_ms.min():.1f}/{np.median(y_ms):.1f}/{y_ms.max():.1f}")

    order = np.argsort(y_ms)  # ascending = faster/better
    top = int(args.top) if args.top else len(order)

    # Optional LLM re-rank over top-K
    if args.llm_model and args.llm_topk and args.llm_topk > 0:
        # Adaptive K: if confident, use smaller topK to shrink prompt
        K_full = int(args.llm_topk)
        K = min(K_full, len(order))  # initial cap; we may shrink after gap calc
        if len(order) > 1:
            top1 = float(y_ms[order[0]])
            top2 = float(y_ms[order[1]])
        else:
            top1 = top2 = float(y_ms[order[0]])
        gap_pct = (top2 - top1) / max(1e-6, top1)
        hazard = bool(ctx[2] > 0.5 or ctx[3] > 0.5)
        K = min((K_full if gap_pct < args.llm_gap_pct else max(1, int(args.llm_topk_confident))), len(order))
        pool = [int(i) for i in order[:K]]
        call_llm = _should_call_llm(
            gap_pct=gap_pct, hazard=hazard,
            top1_ms=top1, top2_ms=top2,
            allow_when_hazard=args.llm_when_hazard,
            ema_llm_ms=_EMA_LLM_MS,
            gain_vs_cost_ratio=float(args.llm_gain_cost_ratio),
            hazard_gap_relax=float(args.llm_hazard_gap_relax),
        )
        llm_overhead_ms = 0.0
        if call_llm:
            # Cache lookup
            ck = _llm_cache_key(ctx, pool)
            cached = _LLM_CACHE.get(ck)
            if cached is not None and len(cached) == 0:
                new = None  # known skip
            else:
                new = cached
            if new is None:
                pool_cands = [
                    {"index": int(i), "pred_ms": float(y_ms[i]),
                     "steps": _steps_from_parsed_plan(good[i]["parsed_plan"])}
                    for i in pool
                ]
                new, llm_overhead_ms = _llm_rerank(ctx, pool_cands, model=args.llm_model, timeout=args.llm_timeout)
                if llm_overhead_ms > 0:
                    _EMA_LLM_MS = _EMA_ALPHA * llm_overhead_ms + (1 - _EMA_ALPHA) * _EMA_LLM_MS
                # Cache result (including "no change" as empty list sentinel)
                if new:
                    _LLM_CACHE[ck] = new
                else:
                    _LLM_CACHE[ck] = []
            elif new:
                pass  # use cached order
            else:
                # cached skip: keep fusion order
                call_llm = False
            if new:
                rest = [i for i in order if int(i) not in set(new)]
                order = np.array(new + [int(i) for i in rest], dtype=int)
                print(f"[score][llm] reranked top-{K} via {args.llm_model}: first={order[0]}  (gap={gap_pct*100:.1f}%  hazard={int(hazard)}  llm_ms={llm_overhead_ms:.0f})")

    rows = []
    for rank, idx in enumerate(order[:top]):
        rows.append({
            "rank": rank,
            "index": int(idx),
            "pred_ms": float(y_ms[idx]),
            "plan": good[idx].get("parsed_plan"),
            "raw":  good[idx].get("raw"),
        })
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"[score] wrote {len(rows)} rows → {args.out}")
    else:
        print(json.dumps(rows, indent=2))

def cmd_deploy(args: argparse.Namespace) -> None:
    global _EMA_LLM_MS
    device = _device_from_arg(args.device)
    model, scaler = load_fusion(Path(args.fusion_dir), device)

    imc = np.load(args.imc_vectors).astype(np.float32)
    good = _load_good(args.good_json)

    # Dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = _maybe_init_dataset(Path(args.dataset_out), args.maybe_wipe)

    rooms = int(args.rooms)
    per_room = int(args.per_room)
    diversity = bool(args.diversity)

    selection_record: Dict[str, Any] = {"rooms": rooms, "per_room": per_room, "selections": [], "llm_total_overhead_ms": 0.0}
    total_ms = 0
    total_plans = 0

    for r in range(rooms):
        # Build context for this room
        room = RoomProfile(r)
        ctx = _compose_ctx_from_room(room)

        # Score all candidates for this room
        y_ms = score_all_for_context(model, scaler, ctx, imc, device)

        # Rank
        order = np.argsort(y_ms)
        sel: List[int] = []

        # (1) LLM picks the first among fusion top-K, if enabled
        llm_overhead_ms = 0.0
        used_llm = False
        if args.llm_model and args.llm_topk and args.llm_topk > 0:
            K_full = int(args.llm_topk)
            pool = [int(i) for i in order[:K_full]]
            if len(order) > 1:
                top1 = float(y_ms[order[0]])
                top2 = float(y_ms[order[1]])
            else:
                top1 = top2 = float(y_ms[order[0]])
            gap_pct = (top2 - top1) / max(1e-6, top1)
            hazard = bool(ctx[2] > 0.5 or ctx[3] > 0.5)
            K = min((K_full if gap_pct < args.llm_gap_pct else max(1, int(args.llm_topk_confident))), len(order))
            pool = [int(i) for i in order[:K]]
            call_llm = _should_call_llm(
                gap_pct=gap_pct, hazard=hazard,
                top1_ms=top1, top2_ms=top2,
                allow_when_hazard=args.llm_when_hazard,
                ema_llm_ms=_EMA_LLM_MS,
                gain_vs_cost_ratio=float(args.llm_gain_cost_ratio),
                hazard_gap_relax=float(args.llm_hazard_gap_relax),
            )
            new = None
            if call_llm:
                ck = _llm_cache_key(ctx, pool)
                cached = _LLM_CACHE.get(ck)
                if cached is not None and len(cached) == 0:
                    new = None  # known skip
                    call_llm = False
                else:
                    new = cached
                if new is None and call_llm:
                    pool_cands = [
                        {"index": int(i), "pred_ms": float(y_ms[i]), "steps": _steps_from_parsed_plan(good[i]["parsed_plan"])}
                        for i in pool
                    ]
                    new, llm_overhead_ms = _llm_rerank(ctx, pool_cands, model=args.llm_model, timeout=args.llm_timeout)
                    if llm_overhead_ms > 0:
                        _EMA_LLM_MS = _EMA_ALPHA * llm_overhead_ms + (1 - _EMA_ALPHA) * _EMA_LLM_MS
                    if new:
                        _LLM_CACHE[ck] = new
                    else:
                        _LLM_CACHE[ck] = []  # sentinel: skip next time
            if new:
                sel.append(int(new[0])); used_llm = True
                selection_record["llm_total_overhead_ms"] += llm_overhead_ms
                print(f"[deploy][llm] room {r:03d} first pick via {args.llm_model}: idx={sel[-1]} (pred_ms={y_ms[sel[-1]]:.1f}, gap={gap_pct*100:.1f}%, hazard={int(hazard)}, llm_ms={llm_overhead_ms:.0f})")
            elif not call_llm:
                # Record skip to avoid repeated attempts on same signature
                ck = _llm_cache_key(ctx, pool)
                _LLM_CACHE.setdefault(ck, [])

        # (2) Fill the remaining slots by fusion (optionally diverse)
        need = max(0, per_room - len(sel))
        if need > 0:
            remaining = [int(i) for i in order if int(i) not in set(sel)]
            if diversity and need > 1:
                pool = remaining[: min(200, len(remaining))]
                V = imc[pool]
                # ensure we pick `need` more (k-center on remaining pool)
                sel_rel = k_center_greedy(V, need, seed=r)
                sel.extend([int(pool[i]) for i in sel_rel])
            else:
                sel.extend(remaining[:need])

        # Build plan steps
        plans = []
        select_items = []
        for idx in sel:
            rec = good[idx]
            steps = _steps_from_parsed_plan(rec["parsed_plan"])
            plans.append(steps)
            select_items.append({
                "index": int(idx),
                "pred_ms": float(y_ms[idx]),
                "raw": rec.get("raw"),
                "parsed_plan": rec.get("parsed_plan"),
            })

        ms, n = _append_episode(ds, r, plans)
        total_ms += ms; total_plans += n
        selection_record["selections"].append({
            "room": r,
            "ctx": ctx.tolist(),
            "chosen": select_items,
            "episode_time_ms": ms,
            "llm_used": bool(used_llm),
            "llm_overhead_ms": float(llm_overhead_ms),
        })
        print(f"[deploy] room {r:03d}: wrote {n} plans, {ms:.0f} ms")

    # Persist alignment
    out_dir = Path(args.dataset_out)
    (out_dir / "runtime_selection.json").write_text(json.dumps(selection_record, indent=2))
    # Also a compact index like your other script for convenience
    executed_index = {
        "dataset_root": str(args.dataset_out),
        "records": [
            {"room": s["room"], "selected_indices": [c["index"] for c in s["chosen"]]}
            for s in selection_record["selections"]
        ],
    }
    (out_dir / "executed_index.json").write_text(json.dumps(executed_index, indent=2))
    print(f"[deploy] done. total plans={total_plans}  total time_ms={total_ms:.0f}")
    print(f"[deploy] wrote selection → {out_dir/'runtime_selection.json'}")
    print(f"[deploy] wrote alignment → {out_dir/'executed_index.json'}")

    # Optional: rebuild matrices
    if args.matrix_builder_module and args.matrix_builder_fn and args.matrix_out:
        try:
            mod = importlib.import_module(args.matrix_builder_module)
            fn = getattr(mod, args.matrix_builder_fn)
            print(f"[deploy] Calling matrix builder: {args.matrix_builder_module}.{args.matrix_builder_fn} (...)")
            fn(dataset_root=str(args.dataset_out), out_dir=str(args.matrix_out))
            print(f"[deploy] Matrix written to {args.matrix_out}")
        except Exception as e:
            print(f"[deploy][warn] Matrix builder failed: {e}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Runtime loop: score + deploy fusion model on rewrites.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    aps = sub.add_parser("score", help="Score all candidates for one context.")
    aps.add_argument("--fusion_dir", type=Path, required=True,
                     help="Directory with best_model.pt (+ scaler.npz or scaler.json).")
    aps.add_argument("--imc_vectors", type=Path, required=True, help="Path to composites.npy.")
    aps.add_argument("--good_json", type=Path, required=True, help="Path to good_rewrites.json.")
    aps.add_argument("--ctx", type=str, default=None, help="size,clutter,heat,gas,noise (comma-separated). If omitted, samples a RoomProfile.")
    aps.add_argument("--top", type=int, default=20, help="How many to output.")
    aps.add_argument("--out", type=Path, default=None, help="Write ranked JSON to file (else prints).")
    aps.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    aps.add_argument("--llm_model", type=str, default="gemma3n:e4b",
                     help="Gemma 3n model tag for `ollama run`, e.g., gemma3n:e2b or gemma3n:e4b.")
    aps.add_argument("--llm_topk", type=int, default=0, help="If >0, re-rank the fusion top-K with LLM.")
    aps.add_argument("--llm_gap_pct", type=float, default=0.10, help="Only call LLM if (top2-top1)/top1 < this.")
    aps.add_argument("--llm_when_hazard", action="store_true", help="Always allow LLM when heat/gas present.")
    aps.add_argument("--llm_timeout", type=float, default=25.0, help="Seconds before giving up on LLM.")
    aps.add_argument("--llm_cache", type=Path, default=None, help="Optional JSON cache file to load/save reranks.")
    aps.add_argument("--llm_topk_confident", type=int, default=3, help="Use smaller topK when fusion is confident (gap >= llm_gap_pct).")
    aps.add_argument("--llm_gain_cost_ratio", type=float, default=0.4, help="Min expected gain vs EMA LLM ms to justify calling (0.4 => need ~40% of EMA).")
    aps.add_argument("--llm_hazard_gap_relax", type=float, default=0.15, help="Relax gap threshold when hazard by this absolute amount.")

    apd = sub.add_parser("deploy", help="Score per-room, pick top-K (optionally diverse), and execute dataset.")
    apd.add_argument("--fusion_dir", type=Path, required=True)
    apd.add_argument("--imc_vectors", type=Path, required=True)
    apd.add_argument("--good_json", type=Path, required=True)
    apd.add_argument("--rooms", type=int, default=40)
    apd.add_argument("--per_room", type=int, default=3)
    apd.add_argument("--diversity", action="store_true", help="Apply k-center to top pool for diversity.")
    apd.add_argument("--dataset_out", type=Path, default=Path("runtime_ds"))
    apd.add_argument("--maybe_wipe", action="store_true")
    apd.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    apd.add_argument("--llm_model", type=str, default="gemma3n:e4b",
                     help="Gemma 3n model tag for `ollama run`, e.g., gemma3n:e2b or gemma3n:e4b.")
    apd.add_argument("--llm_topk", type=int, default=0, help="If >0, LLM re-ranks fusion top-K; first is taken.")
    apd.add_argument("--llm_gap_pct", type=float, default=0.10, help="Only call LLM if (top2-top1)/top1 < this.")
    apd.add_argument("--llm_when_hazard", action="store_true", help="Always allow LLM when heat/gas present.")
    apd.add_argument("--llm_timeout", type=float, default=25.0, help="Seconds before giving up on LLM.")
    apd.add_argument("--llm_cache", type=Path, default=None, help="Optional JSON cache file to load/save reranks.")
    apd.add_argument("--llm_topk_confident", type=int, default=3, help="Use smaller topK when fusion is confident (gap >= llm_gap_pct).")
    apd.add_argument("--llm_gain_cost_ratio", type=float, default=0.4, help="Min expected gain vs EMA LLM ms to justify calling (0.4 => need ~40% of EMA).")
    apd.add_argument("--llm_hazard_gap_relax", type=float, default=0.15, help="Relax gap threshold when hazard by this absolute amount.")

    apd.add_argument("--matrix_builder_module", type=str, default=None)
    apd.add_argument("--matrix_builder_fn", type=str, default=None)
    apd.add_argument("--matrix_out", type=Path, default=None)

    return ap

def main():
    ap = build_cli()
    args = ap.parse_args()
    # Load cache if provided
    if getattr(args, "llm_cache", None):
        p = args.llm_cache
        if p and Path(p).exists():
            try:
                data = _json.loads(Path(p).read_text())
                if isinstance(data, dict):
                    # keys are hex strings → lists of ints
                    _LLM_CACHE.update({k: v for k, v in data.items() if isinstance(v, list)})
                    print(f"[llm][cache] loaded {len(_LLM_CACHE)} entries from {p}")
            except Exception as e:
                print(f"[llm][cache][warn] failed to load {p}: {e}")
    if args.cmd == "score":
        cmd_score(args)
    elif args.cmd == "deploy":
        cmd_deploy(args)
    else:
        raise ValueError(f"Unknown cmd={args.cmd}")
    # Save cache if provided
    if getattr(args, "llm_cache", None):
        p = args.llm_cache
        if p:
            try:
                Path(p).write_text(_json.dumps(_LLM_CACHE, separators=(",", ":"), indent=0))
                print(f"[llm][cache] saved {len(_LLM_CACHE)} entries → {p}")
            except Exception as e:
                print(f"[llm][cache][warn] failed to save {p}: {e}")

if __name__ == "__main__":
    main()

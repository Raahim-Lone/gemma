#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_and_execute.py

Subcommands
-----------
select:
  Perform greedy farthest-point (k-center) on composites.npy (L2) and
  write selected indices to selected_ids.json.

execute:
  Load selected_ids.json + good_rewrites.json, convert each plan to
  primitive action lists, simulate them (reusing run_skill & logging schema),
  and append to a LeRobotDataset. Optionally call a matrix-builder function.

Examples
--------
# 1) Select 120 diverse composites
python select_and_execute.py select \
  --composites imc_out/composites.npy \
  --k 120 \
  --out imc_out/selected_ids.json \
  --seed 42

# 2) Execute them into a dataset and (optionally) update a cost matrix
python select_and_execute.py execute \
  --good_json imc_out/good_rewrites.json \
  --selected imc_out/selected_ids.json \
  --rooms 50 \
  --dataset_out lerobot_exec_ds \
  --maybe_wipe \
  --matrix_builder_module export_cost_matrix \
  --matrix_builder_fn build_from_dataset \
  --matrix_out matrix_v3
"""
from __future__ import annotations
import argparse, importlib, json, math, os, random, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Compatibility with your simulation/logger
# We try to import your helpers from log_sim_le_robot.py.
# If not available, we provide safe fallbacks with the same interface.
# -----------------------------------------------------------------------------
RunSkillFn = None
FEATURES: Dict[str, Dict[str, Any]] = {}

def _try_import_sim():
    global RunSkillFn, FEATURES, RoomProfile, BatterySim
    try:
        # Your file name in the previous message was: log_sim_le_robot.py
        sim = importlib.import_module("log_sim_le_robot")
        RunSkillFn = sim.run_skill
        FEATURES = sim.FEATURES
        RoomProfile = sim.RoomProfile
        BatterySim = sim.BatterySim
        print("[execute] Using run_skill/FEATURES from log_sim_le_robot.py")
        return True
    except Exception as e:
        print(f"[execute][warn] Could not import log_sim_le_robot.py ({e}). Using built-in fallbacks.")
        return False

_fallback_loaded = _try_import_sim()

# --- Fallback minimal sim (mirrors your previous definitions) -----------------
if not _fallback_loaded:
    import random, time
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

    import numpy as _np
    def _sleep_noise(dt: float):
        time.sleep(max(0.0, dt))

    def _lidar_points(room: RoomProfile, dt: float):
        pts = _np.random.normal(2.0, 0.3, 800)
        short = _np.random.rand(800) < room.clutter * 0.4
        pts[short] *= 0.25
        _sleep_noise(dt);   return pts

    def _thermal_frame(room: RoomProfile, dt: float):
        frame = _np.random.normal(23, 1, (8, 8))
        if room.heat:
            frame.flat[_np.random.randint(0, 64)] = 48 + _np.random.rand() * 5
        _sleep_noise(dt);   return frame

    def _gas_ppm(room: RoomProfile, dt: float):
        _sleep_noise(dt)
        return _np.random.normal(200, 20) + (350 if room.gas else 0)

    def _audio_rms(room: RoomProfile, dt: float):
        _sleep_noise(dt)
        return _np.random.normal(0.03, 0.01) + (0.08 if room.noise else 0)

    def _run_skill_fallback(name: str, room: RoomProfile, battery: BatterySim, *, parallel_scale: float = 1.0) -> Dict[str, Any]:
        start = time.time()
        batt0 = battery.read()
        if   name == "wait":
            _sleep_noise(0.6 * parallel_scale)
            hazard = 0; safety = 0; watts = 1; dt = 0.6 * parallel_scale
        elif name == "lidar_scan":
            pts = _lidar_points(room, 2.0 * parallel_scale)
            hazard = int(min(pts) < 0.3)
            safety = int(random.random() < room.clutter * 0.05)
            watts  = 15; dt = 2.0 * parallel_scale
        elif name == "thermal_snap":
            frame = _thermal_frame(room, 0.8 * parallel_scale)
            hazard = int(frame.max() > 45)
            safety = 0; watts = 5; dt = 0.8 * parallel_scale
        elif name == "gas_sniff":
            ppm = _gas_ppm(room, 1.0 * parallel_scale)
            hazard = int(ppm > 400)
            safety = 0; watts = 8; dt = 1.0 * parallel_scale
        else:  # audio_probe
            rms = _audio_rms(room, 0.5 * parallel_scale)
            hazard = int(rms > 0.08)
            safety = 0; watts = 2; dt = 0.5 * parallel_scale

        battery.drain(watts, dt)
        return dict(
            observation     = _np.zeros((1,), dtype=_np.float32),
            action          = name,
            reward          = _np.array([-dt], dtype=_np.float32),
            time_ms         = _np.array([int(dt * 1000)], dtype=_np.int64),
            energy_mWh      = _np.array([batt0 - battery.read()], dtype=_np.float32),
            hazard_found    = _np.array([bool(hazard)]),
            safety_incident = _np.array([bool(safety)]),
            room_size       = _np.array([room.size_m2], dtype=_np.float32),
            room_clutter    = _np.array([room.clutter], dtype=_np.float32),
            room_heat       = _np.array([1.0 if room.heat else 0.0], dtype=_np.float32),
            room_gas        = _np.array([1.0 if room.gas else 0.0], dtype=_np.float32),
            room_noise      = _np.array([1.0 if room.noise else 0.0], dtype=_np.float32),
        )
    RunSkillFn = _run_skill_fallback

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
# Primitive alias map (rewrite_pipeline used 'movement_pause'; sim uses 'wait')
# -----------------------------------------------------------------------------
PRIMITIVE_ALIAS = {
    "movement_pause": "wait",
    "pause": "wait",
    "idle": "wait",
    "noop": "wait",
}

ALLOWED_PRIMS = {"lidar_scan","thermal_snap","gas_sniff","audio_probe","wait"}

def _normalize_primitive(p: str) -> str:
    p = p.strip()
    p = PRIMITIVE_ALIAS.get(p, p)
    if p not in ALLOWED_PRIMS:
        raise ValueError(f"Unknown primitive after normalization: {p}")
    return p

# -----------------------------------------------------------------------------
# SELECT: Greedy k-center on L2 distances
# -----------------------------------------------------------------------------
def k_center_greedy(M: np.ndarray, k: int, *, seed: int = 0, init_idx: Optional[int] = None) -> List[int]:
    """
    Greedy farthest-point selection.
    Assumes rows are feature vectors (optionally unit-normalized).
    """
    n = M.shape[0]
    if k <= 0 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}")
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Choose initial center
    if init_idx is None:
        init_idx = rng.randrange(n)
    selected = [init_idx]

    # Track min distance to the current selected set for each point
    dmin = np.linalg.norm(M - M[init_idx:init_idx+1], axis=1)

    while len(selected) < k:
        # pick the point with max min-distance to current centers
        cand = int(np.argmax(dmin))
        selected.append(cand)
        # update dmin
        dnew = np.linalg.norm(M - M[cand:cand+1], axis=1)
        dmin = np.minimum(dmin, dnew)
    return selected

def cmd_select(args: argparse.Namespace) -> None:
    M = np.load(args.composites)
    if M.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {M.shape}")
    print(f"[select] composites: shape={M.shape}  (row-normalized={args.assume_row_normalized})")

    # If not normalized, you can still run k-center on raw L2.
    sels = k_center_greedy(M, args.k, seed=args.seed, init_idx=args.init)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(sels, indent=2))
    print(f"[select] wrote {len(sels)} indices → {args.out}")

# -----------------------------------------------------------------------------
# EXECUTE: simulate the selected composites into a LeRobot dataset
# -----------------------------------------------------------------------------
def _load_good(good_json: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(good_json).read_text())
    if not isinstance(data, list):
        raise ValueError("good_rewrites.json must contain a list")
    return data

def _load_selected(selected_json: Path) -> List[int]:
    data = json.loads(Path(selected_json).read_text())
    if not (isinstance(data, list) and all(isinstance(x, int) for x in data)):
        raise ValueError("selected_ids.json must be a list of integers")
    return data

def _steps_from_parsed_plan(parsed_plan: List[List[str]]) -> List[List[str]]:
    """
    Convert rewrite_pipeline parsed_plan (list of list of primitives, parallel if len>1)
    to normalized primitive steps (mapping movement_pause -> wait).
    """
    steps: List[List[str]] = []
    for step in parsed_plan:
        norm = [_normalize_primitive(p) for p in step]
        steps.append(norm)
    return steps

def _plan_key_from_steps(steps: List[List[str]]) -> str:
    # canonical: join parallel items with '|', steps with ', '
    return ", ".join("|".join(step) for step in steps)

def _maybe_init_dataset(root: Path, maybe_wipe: bool) -> "LeRobotDataset":
    """
    Create or load a LeRobotDataset at root. If --maybe_wipe is set and root exists,
    it is removed before creation.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if root.exists() and maybe_wipe:
        import shutil
        shutil.rmtree(root)
    if root.exists():
        try:
            ds = LeRobotDataset.load(root=str(root))
            print(f"[execute] Loaded existing LeRobotDataset at {root}")
            return ds
        except Exception:
            # If load fails, recreate
            import shutil
            shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id="exec_rooms", root=str(root), fps=10, features=FEATURES)
    print(f"[execute] Created new LeRobotDataset at {root}")
    return ds

def _append_episode(ds: "LeRobotDataset", room_idx: int, plans: List[List[List[str]]]) -> Tuple[int,int]:
    """
    Append one episode for a room, executing all given plans.
    Returns (total_time_ms, num_plans).
    """
    # Meta frame (matches your logger)
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
    total_episode_ms = 0
    for parsed in plans:
        # parsed is List[List[str]] steps; len>1 => parallel group
        # compute simple stats for summary fields
        flat = [p for step in parsed for p in step]
        num_nodes = len(flat)
        num_parallel = sum(1 for step in parsed if len(step) > 1)
        depth = 2 if num_parallel > 0 else 1

        pid = plan_counter; plan_counter += 1
        comp_ms, comp_mWh, any_safe, any_haz = 0.0, 0.0, False, False

        # Build a mapping step_index -> group_id (adjacent pairs → group id)
        # Here each parallel step is one group; group id increments for each such step.
        gid_counter = -1

        node_counter = 0
        for step_idx, step in enumerate(parsed):
            is_parallel = len(step) > 1
            gid = -1
            if is_parallel:
                gid_counter += 1
                gid = gid_counter

            # Execute each primitive in the step (separately), but shorten dt via scale
            scale = 0.7 if is_parallel else 1.0
            for prim in step:
                frame = RunSkillFn(prim, room, battery, parallel_scale=scale)

                ds.add_frame(
                    dict(
                        **frame,
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

                comp_ms  += float(frame["time_ms"][0])
                comp_mWh += float(frame["energy_mWh"][0])
                any_safe |= bool(frame["safety_incident"][0])
                any_haz  |= bool(frame["hazard_found"][0])
                node_counter += 1

        # summary row
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

def cmd_execute(args: argparse.Namespace) -> None:
    good = _load_good(args.good_json)
    sel  = _load_selected(args.selected)

    # Build the plan list in the selection order
    plans: List[List[List[str]]] = []
    plan_keys: List[str] = []
    for idx in sel:
        if idx < 0 or idx >= len(good):
            raise IndexError(f"selected index {idx} out of range for good_rewrites.json (N={len(good)})")
        rec = good[idx]
        parsed = rec.get("parsed_plan")
        if not isinstance(parsed, list):
            raise ValueError(f"Record {idx} missing parsed_plan")
        steps = _steps_from_parsed_plan(parsed)
        plans.append(steps)
        plan_keys.append(_plan_key_from_steps(steps))

    # Prepare dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = _maybe_init_dataset(Path(args.dataset_out), args.maybe_wipe)

    rng = random.Random(args.seed)
    rooms = args.rooms
    per_room = max(1, math.ceil(len(plans) / rooms))
    print(f"[execute] rooms={rooms}  total_plans={len(plans)}  ~{per_room} plans/room")
    used_rooms = min(rooms, math.ceil(len(plans) / per_room))  # e.g., 120 plans / 3 per room = 40 rooms
    print(f"[execute] will actually create {used_rooms} episode(s) ({per_room} plan(s)/room)")

    # Assign plans to rooms (round-robin chunking)
    total_plans_written = 0
    total_ms = 0
    exec_index: Dict[str, List[Dict[str, Any]]] = {}
    for r in range(used_rooms):
        start = r * per_room
        end   = min((r+1) * per_room, len(plans))
        ms, n = _append_episode(ds, r, plans[start:end])
        total_plans_written += n
        total_ms += ms
        print(f"[execute] room {r:03d}: wrote {n} plans, {ms:.0f} ms")
        # plan_id resets per episode (0..n-1). Record IMC index & key for each.
        recs: List[Dict[str, Any]] = []
        for j, imc_global_idx in enumerate(sel[start:end]):
            recs.append({
                "pid": j,  # plan_id within this episode
                "imc_idx": int(imc_global_idx),
                "plan_key": plan_keys[start + j]
            })
        exec_index[f"room_{r:03d}"] = recs

    print(f"[execute] done. total plans={total_plans_written}  total time_ms={total_ms:.0f}")
    (Path(args.dataset_out) / "executed_index.json").write_text(json.dumps(exec_index, indent=2))
    print(f"[execute] wrote alignment → {Path(args.dataset_out) / 'executed_index.json'}")

    # Optional post-hook: call a matrix builder if provided
    if args.matrix_builder_module and args.matrix_builder_fn and args.matrix_out:
        try:
            mod = importlib.import_module(args.matrix_builder_module)
            fn = getattr(mod, args.matrix_builder_fn)
            print(f"[execute] Calling matrix builder: {args.matrix_builder_module}.{args.matrix_builder_fn} (...)")
            fn(dataset_root=str(args.dataset_out), out_dir=str(args.matrix_out))
            print(f"[execute] Matrix written to {args.matrix_out}")
        except Exception as e:
            print(f"[execute][warn] Matrix builder failed: {e}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Select and Execute composite plans.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    aps = sub.add_parser("select", help="Greedy k-center selection on composites.npy")
    aps.add_argument("--composites", type=Path, required=True)
    aps.add_argument("--k", type=int, required=True)
    aps.add_argument("--out", type=Path, required=True)
    aps.add_argument("--seed", type=int, default=0)
    aps.add_argument("--init", type=int, default=None, help="Optional initial index")
    aps.add_argument("--assume_row_normalized", action="store_true",
                     help="For info only; we use L2 regardless.")

    ape = sub.add_parser("execute", help="Execute selected plans into a LeRobot dataset.")
    ape.add_argument("--good_json", type=Path, required=True)
    ape.add_argument("--selected", type=Path, required=True)
    ape.add_argument("--rooms", type=int, default=50)
    ape.add_argument("--dataset_out", type=Path, default=Path("lerobot_exec_ds"))
    ape.add_argument("--maybe_wipe", action="store_true", help="If set, recreate dataset_out each run.")
    ape.add_argument("--seed", type=int, default=0)

    # Optional post-processing hook for matrix building
    ape.add_argument("--matrix_builder_module", type=str, default=None,
                     help="e.g., 'build_cost_matrix'")
    ape.add_argument("--matrix_builder_fn", type=str, default=None,
                     help="e.g., 'build_from_dataset'")
    ape.add_argument("--matrix_out", type=Path, default=None)

    return ap

def main():
    ap = build_cli()
    args = ap.parse_args()
    if args.cmd == "select":
        cmd_select(args)
    elif args.cmd == "execute":
        cmd_execute(args)
    else:
        raise ValueError(f"Unknown cmd={args.cmd}")

if __name__ == "__main__":
    main()

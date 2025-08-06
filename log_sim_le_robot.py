"""
log_sim_le_robot.py
Fully‑working simulation‑only logger that writes a REAL LeRobotDataset.
python le_robot.py --rooms 50 --out lerobot_ds

"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, Any
import shutil
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset   # API is still v1.x

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
     # ── NEW: composite/plan metadata per frame ───────────────────────────
     "plan_id":         {"dtype": "int64",   "shape": (1,), "names": None},
     "plan_node":       {"dtype": "int64",   "shape": (1,), "names": None},  # index within plan
     "parallel_group":  {"dtype": "int64",   "shape": (1,), "names": None},  # -1 if not in parallel
     "is_summary":      {"dtype": "bool",    "shape": (1,), "names": None},  # True only for summary row
     # small, fixed graph summary features (repeat on every node frame for simplicity)
     "comp_num_nodes":  {"dtype": "int64",   "shape": (1,), "names": None},
     "comp_num_parallel":{"dtype": "int64",  "shape": (1,), "names": None},
     "comp_depth":      {"dtype": "int64",   "shape": (1,), "names": None},

 }


# -----------------------------------------------------------------------------
# Sim‑world helpers (same as before, trimmed for brevity)
# -----------------------------------------------------------------------------
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

def lidar_points(room: RoomProfile, dt: float):
    pts = np.random.normal(2.0, 0.3, 800)
    short = np.random.rand(800) < room.clutter * 0.4
    pts[short] *= 0.25
    time.sleep(dt);   return pts

def thermal_frame(room: RoomProfile, dt: float):
    frame = np.random.normal(23, 1, (8, 8))
    if room.heat:
        frame.flat[np.random.randint(0, 64)] = 48 + np.random.rand() * 5
    time.sleep(dt);   return frame

def gas_ppm(room: RoomProfile, dt: float):
    time.sleep(dt)
    return np.random.normal(200, 20) + (350 if room.gas else 0)

def audio_rms(room: RoomProfile, dt: float):
    time.sleep(dt)
    return np.random.normal(0.03, 0.01) + (0.08 if room.noise else 0)

# -----------------------------------------------------------------------------
# Primitive execution returning a *step dict* LeRobot likes
# -----------------------------------------------------------------------------
def run_skill(name: str, room: RoomProfile, battery: BatterySim, *, parallel_scale: float = 1.0) -> Dict[str, Any]:
    start = time.time()
    batt0 = battery.read()
    if   name == "wait":
        # idle step: consumes little power, adds time; no hazard/safety
        time.sleep(0.6 * parallel_scale)
        hazard = 0; safety = 0; watts = 1; dt = 0.6 * parallel_scale
    elif name == "lidar_scan":
        pts = lidar_points(room, 2.0)
        hazard = int(min(pts) < 0.3)
        safety = int(random.random() < room.clutter * 0.05)
        watts  = 15
    elif name == "thermal_snap":
        frame = thermal_frame(room, 0.8)
        hazard = int(frame.max() > 45)
        safety = 0
        watts  = 5
    elif name == "gas_sniff":
        ppm = gas_ppm(room, 1.0)
        hazard = int(ppm > 400)
        safety = 0
        watts  = 8
    else:  # audio_probe
        rms = audio_rms(room, 0.5)
        hazard = int(rms > 0.08)
        safety = 0
        watts  = 2

    dt = (time.time() - start) * parallel_scale
    battery.drain(watts, dt)
    return dict(
     observation     = np.zeros((1,), dtype=np.float32),
     action          = name,
     reward          = np.array([-dt], dtype=np.float32),
     time_ms         = np.array([int(dt * 1000)], dtype=np.int64),
     energy_mWh      = np.array([batt0 - battery.read()], dtype=np.float32),
     hazard_found    = np.array([bool(hazard)]),
     safety_incident = np.array([bool(safety)]),
     room_size       = np.array([room.size_m2], dtype=np.float32),
     room_clutter    = np.array([room.clutter], dtype=np.float32),
     room_heat       = np.array([1.0 if room.heat else 0.0], dtype=np.float32),
     room_gas        = np.array([1.0 if room.gas else 0.0], dtype=np.float32),
     room_noise      = np.array([1.0 if room.noise else 0.0], dtype=np.float32),
 )

PRIMITIVES = ["lidar_scan", "thermal_snap", "gas_sniff", "audio_probe"]
# ─────────────────────────────────────────────────────────────────────────────
# Composite plan sampling helpers
# ─────────────────────────────────────────────────────────────────────────────
# knobs (feel free to tweak)
COMPOSITES_PER_ROOM = 16
MIN_LEN, MAX_LEN    = 2, 5
PARALLEL_PROB       = 0.35     # chance that a random adjacent pair becomes parallel
DUPLICATE_PROB      = 0.20     # inject a duplicate (inefficiency)
WAIT_PROB           = 0.20     # inject a "wait" (idle) node (inefficiency)

def sample_composite_plan():
    """Return (nodes, parallel_groups) where:
       nodes = list of primitives (strings)
       parallel_groups = dict{group_id: [node_indices]} ; group_id >=0
       Nodes not in a group implicitly run sequentially.
    """
    k   = random.randint(MIN_LEN, MAX_LEN)
    seq = random.choices(PRIMITIVES, k=k)
    # inefficiency injections
    if random.random() < DUPLICATE_PROB:
        pos = random.randrange(len(seq))
        seq.insert(pos, seq[pos])   # duplicate a step
    if random.random() < WAIT_PROB:
        pos = random.randrange(len(seq))
        seq.insert(pos, "wait")     # synthetic idle step (adds time only)
    # simple parallel grouping on adjacent pair
    parallel = {}
    gid = -1
    i = 0
    while i < len(seq) - 1:
        if seq[i] != "wait" and seq[i+1] != "wait" and random.random() < PARALLEL_PROB:
            gid += 1
            parallel[gid] = [i, i+1]
            i += 2
        else:
            i += 1
    # depth is 2 if any parallel pair, else 1 (toy metric)
    depth = 2 if parallel else 1
    num_parallel = len(parallel)
    return seq, parallel, depth, len(seq), num_parallel


# -----------------------------------------------------------------------------
# Main routine: build dataset with EpisodeBuilder
# -----------------------------------------------------------------------------
def main(n_rooms: int, out_dir: Path):
    # LeRobotDataset.create() insists the root dir not exist
    if out_dir.exists():                      # wipe an earlier run
        shutil.rmtree(out_dir)
    ds = LeRobotDataset.create(
        repo_id="sim_rooms",
        root=str(out_dir),
        fps=10,
        features=FEATURES,
    )
    plan_counter = 0
    for idx in range(n_rooms):
        room = RoomProfile(idx)
        battery = BatterySim()
        # ---- start a new episode ------------------------------------------------
        ds.clear_episode_buffer()                   # wipe internal buffer
        ds.add_frame(                              # meta-frame first
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
            task="simulation",        # ← required positional arg
        )

        # add one frame per primitive skill
        # ── NEW: generate several composite plans per room ───────────────
        for _ in range(COMPOSITES_PER_ROOM):
            nodes, parallel, depth, num_nodes, num_parallel = sample_composite_plan()
            pid = plan_counter; plan_counter += 1

            total_ms = 0.0
            total_mWh= 0.0
            any_safe = False
            any_haz  = False

            for ni, prim in enumerate(nodes):
                # parallel speedup if node in a parallel group
                pg = -1
                for gid, idxs in parallel.items():
                    if ni in idxs:
                        pg = gid; break
                # toy parallel scaling factor (two nodes share time)
                scale = 0.7 if pg >= 0 else 1.0
                frame = run_skill(prim, room, battery, parallel_scale=scale)
                frame["plan_id"]        = np.array([pid], dtype=np.int64)
                frame["plan_node"]      = np.array([ni], dtype=np.int64)
                frame["parallel_group"] = np.array([pg], dtype=np.int64)
                frame["is_summary"]     = np.array([False])
                frame["comp_num_nodes"] = np.array([num_nodes], dtype=np.int64)
                frame["comp_num_parallel"] = np.array([num_parallel], dtype=np.int64)
                frame["comp_depth"]     = np.array([depth], dtype=np.int64)
                ds.add_frame(frame, task="simulation")
                # accumulate composite metrics
                total_ms  += float(frame["time_ms"][0])
                total_mWh += float(frame["energy_mWh"][0])
                any_safe  |= bool(frame["safety_incident"][0])
                any_haz   |= bool(frame["hazard_found"][0])

            # add one composite summary row (action="composite_summary")
            summary = dict(
                observation     = np.zeros((1,), dtype=np.float32),
                action          = "composite_summary",
                reward          = np.array([-(total_ms/1000.0)], dtype=np.float32),
                time_ms         = np.array([int(total_ms)], dtype=np.int64),
                energy_mWh      = np.array([total_mWh], dtype=np.float32),
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

        ds.save_episode()                           # flush episode to disk
        print(f"✔ logged {room.room_id}")

    print(f"✅ Dataset ready at → {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=20)
    ap.add_argument("--out",   type=Path, default=Path("lerobot_ds"))
    args = ap.parse_args()
    main(args.rooms, args.out)

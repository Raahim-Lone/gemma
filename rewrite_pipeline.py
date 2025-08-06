#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rewrite_pipeline.py
Subcommands:
  - generate: sample seed plans, ask Gemma (via Ollama) for rewrites, save raw outputs
  - filter:   validate + score rewrites, compute composite vectors, save artifacts

Examples
--------
# 1) Generate 2k candidates with Gemma via Ollama:
python rewrite_pipeline.py generate \
  --out_dir imc_out \
  --model gemma2:2b-instruct \
  --n 2000 \
  --batch_size 8 \
  --temperature 0.7 \
  --seed 13

# 2) Filter and embed to composites:
python rewrite_pipeline.py filter \
  --in_dir imc_out \
  --out_dir imc_out \
  --max_nodes 8 \
  --min_nodes 2

Artifacts
---------
imc_out/
  raw/
    rewrites_YYYYmmdd_HHMMSS.jsonl         (append-only raw model generations)
  good_rewrites.json                        (parsed, accepted, scored)
  composites.npy                            (float32, shape [N, D])
  feature_schema.json                       (to document vector meaning)
"""

from __future__ import annotations
import argparse, json, math, os, random, re, subprocess, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import requests
except Exception:
    requests = None  # we will fallback to CLI if requests not available


# -------------------------------
# Domain constants (edit as needed)
# -------------------------------

# Allowed atomic skills / primitives (extend to your project)
PRIMITIVES = [
    "lidar_scan",
    "thermal_snap",
    "gas_sniff",
    "audio_probe",
    "wait",            # exact name used by your run_skill()
]
ALIASES = {
    "movement_pause": "wait",
    "pause": "wait",
    "idle": "wait",
}


def _canon_prim(t: str) -> str:
    return ALIASES.get(t, t)

# Nominal power draw (Watts), rough priors
NOMINAL_WATTS = {
    "lidar_scan":     15.0,
    "thermal_snap":    5.0,
    "gas_sniff":       8.0,
    "audio_probe":     2.0,
    "wait":            1.0,
}

# Nominal time per primitive (milliseconds), rough priors
NOMINAL_MS = {
    "lidar_scan":     2000.0,
    "thermal_snap":    800.0,
    "gas_sniff":      1000.0,
    "audio_probe":     500.0,
    "wait":             600.0,   # your idle step
}

# Safety weight: mark safety-relevant primitives (used for feature includes_safety)
SAFETY_CRIT = {
    "lidar_scan": 1.0,   # you use lidar minima to flag hazards
    "thermal_snap": 0.0,
    "gas_sniff": 0.0,
    "audio_probe": 0.0,
    "wait": 0.0,
}

# -------------------------------
# Simple plan representation
# -------------------------------
# A plan is represented as a list of steps; each step is a list (size >=1) of primitives.
# If len(step) > 1, that step is a parallel group, e.g., ["lidar_scan","thermal_snap"].


def _tokenize_plan_string(s: str) -> List[str]:
    """Split on commas, plus, or within minimal parentheses; keep primitives alnum/_ only."""
    s = s.strip()
    # Normalize delimiters: treat ';' as ','; '|' as '+' to denote parallel.
    s = s.replace(";", ",").replace("||", "+").replace("|", "+")
    # Remove surrounding brackets/quotes if present
    s = s.strip("[](){}").strip()
    # Split on commas for steps
    parts = [p.strip() for p in re.split(r",", s) if p.strip()]
    return parts


def parse_step_to_group(step: str) -> List[str]:
    """
    Parse one step. Accept:
      - 'a'                    -> ['a']
      - 'a + b' or 'a+b'       -> ['a','b'] (parallel)
      - '(a + b)'              -> ['a','b']
    """
    step = step.strip().strip("()[]{}").strip()
    # split on '+' (parallel)
    toks = [t.strip() for t in re.split(r"\+", step) if t.strip()]
    # keep only word-ish tokens
    toks = [re.sub(r"[^a-zA-Z0-9_]", "", t) for t in toks]
    toks = [_canon_prim(t) for t in toks if t]
    return toks


def parse_plan_text(text: str) -> List[List[str]]:
    """
    Try hard to parse model output into plan steps.
    Accepts either JSON or a DSL string (comma-separated steps; '+' for parallel).
    Returns: list of steps, each step is list of primitives (size>=1).
    Raises ValueError if cannot recover a valid plan.
    """
    # 1) Try JSON first
    j = None
    try:
        j = json.loads(text)
    except Exception:
        # Attempt to extract first JSON-looking region
        m = re.search(r"(\[[\s\S]+])", text)
        if m:
            try:
                j = json.loads(m.group(1))
            except Exception:
                j = None

    if isinstance(j, list):
        if all(isinstance(x, str) for x in j):
            parts = [",".join([_canon_prim(re.sub(r"[^a-zA-Z0-9_]", "", x.strip()))]) for x in j]
        elif all(isinstance(x, list) for x in j) and all(
            all(isinstance(y, str) for y in x) for x in j
        ):
            # already list of lists
            groups = []
            for step_list in j:
                step = [_canon_prim(re.sub(r"[^a-zA-Z0-9_]", "", t).strip()) for t in step_list]
                step = [t for t in step if t]
                if step:
                    groups.append(step)
            if groups:
                return groups
            raise ValueError("JSON list-of-lists was empty after cleaning.")
        else:
            # flatten any dicts by 'op' field heuristically
            parts = []
            for x in j:
                if isinstance(x, dict) and "op" in x:
                    parts.append(str(x["op"]))
                else:
                    parts.append(str(x))
    else:
        # 2) fallback to DSL
        parts = _tokenize_plan_string(text)

    # Convert per-step tokens to groups
    groups = []
    for p in parts:
        g = parse_step_to_group(p)
        if g:
            groups.append(g)
    if not groups:
        raise ValueError("Could not parse any valid steps.")
    return groups


def validate_plan(groups: List[List[str]], *, min_nodes: int, max_nodes: int) -> Tuple[bool, str]:
    """Basic checks: allowed primitives, length limits, duplicate spam."""
    # Flatten to primitives
    flat = [a for step in groups for a in step]
    # Allowed
    bad = [a for a in flat if a not in PRIMITIVES]
    if bad:
        return False, f"Unknown primitive(s): {sorted(set(bad))}"

    # Count nodes
    n_nodes = len(flat)
    if n_nodes < min_nodes:
        return False, f"Too short: {n_nodes} < {min_nodes}"
    if n_nodes > max_nodes:
        return False, f"Too long: {n_nodes} > {max_nodes}"

    # Duplicate spam (e.g., same op >50% of nodes)
    if flat:
        most = max(flat, key=flat.count)
        if flat.count(most) / len(flat) > 0.6:
            return False, f"Duplicate spam: '{most}' occurs {flat.count(most)}/{len(flat)}"

    return True, "ok"


def plan_depth(groups: List[List[str]]) -> int:
    """Depth=1 if all serial; 2 if any parallel; (extend if you support nesting)."""
    return 2 if any(len(g) > 1 for g in groups) else 1


def plan_num_parallel(groups: List[List[str]]) -> int:
    """Number of parallel groups (len(step) >= 2)."""
    return sum(1 for g in groups if len(g) >= 2)


def estimate_time_ms(groups: List[List[str]]) -> float:
    """
    Naive time estimate:
      - serial step: sum of NOMINAL_MS
      - parallel step: max of NOMINAL_MS in that step (with a small overhead)
    """
    total = 0.0
    for g in groups:
        if len(g) == 1:
            total += NOMINAL_MS.get(g[0], 800.0)
        else:
            m = max(NOMINAL_MS.get(a, 800.0) for a in g)
            total += m * 1.15  # little overhead for parallel coordination
    return float(total)


def compute_features(groups: List[List[str]]) -> Dict[str, float]:
    flat = [a for step in groups for a in step]
    n_nodes = len(flat)
    n_parallel = plan_num_parallel(groups)
    depth = plan_depth(groups)
    mean_watts = float(np.mean([NOMINAL_WATTS.get(a, 5.0) for a in flat])) if flat else 0.0
    t_ms = estimate_time_ms(groups)
    includes_safety = 1.0 if any(SAFETY_CRIT.get(a, 0.0) > 0.0 for a in flat) else 0.0
    return dict(
        comp_num_nodes=float(n_nodes),
        comp_num_parallel=float(n_parallel),
        comp_depth=float(depth),
        mean_watts=float(mean_watts),
        est_time_ms=float(t_ms),
        includes_safety=float(includes_safety),
    )

def groups_to_composite(groups: List[List[str]]):
    """
    Convert list-of-steps (each a list of primitives; len>1 = parallel) into:
      nodes: flat list of primitives in execution order
      parallel: {gid: [node_indices]}  (gid starts at 0)
      depth, num_nodes, num_parallel: small graph summary
    """
    nodes: List[str] = []
    parallel: Dict[int, List[int]] = {}
    gid = 0
    for step in groups:
        if len(step) == 1:
            nodes.append(step[0])
        else:
            start = len(nodes)
            for prim in step:
                nodes.append(prim)
            parallel[gid] = list(range(start, start + len(step)))
            gid += 1
    depth = 2 if parallel else 1
    num_nodes = len(nodes)
    num_parallel = len(parallel)
    return nodes, parallel, depth, num_nodes, num_parallel


# -------------------------------
# Ollama calling utils
# -------------------------------

def ollama_generate_http(
    model: str, prompt: str, *, temperature: float, seed: Optional[int], host: str
) -> str:
    """
    Call Ollama REST API /api/generate.
    Returns full response text (not streamed). Raises on failure.
    """
    if requests is None:
        raise RuntimeError("requests not installed; cannot use HTTP API.")
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    if seed is not None:
        payload["options"]["seed"] = int(seed)
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Expected field 'response' (single string)
    return data.get("response", "")


def ollama_generate_cli(
    model: str, prompt: str, *, temperature: float, seed: Optional[int]
) -> str:
    """
    Fallback CLI call: `ollama run MODEL -o temperature=... [-o seed=...] -p PROMPT`
    Returns stdout.
    """
    cmd = ["ollama", "run", model, "-o", f"temperature={float(temperature)}", "-p", prompt]
    if seed is not None:
        cmd[3:3] = ["-o", f"seed={int(seed)}"]  # insert extra -o before -p
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return res.stdout


def call_model(
    model: str,
    prompt: str,
    *,
    temperature: float,
    seed: Optional[int],
    host: str = "http://localhost:11434",
    prefer_http: bool = True,
) -> str:
    """Try HTTP; fallback to CLI."""
    if prefer_http:
        try:
            return ollama_generate_http(model, prompt, temperature=temperature, seed=seed, host=host)
        except Exception as e:
            print(f"[warn] HTTP call failed ({e}); falling back to CLI...")
    return ollama_generate_cli(model, prompt, temperature=temperature, seed=seed)


# -------------------------------
# Prompting
# -------------------------------

FEW_SHOT = """You rewrite robot sensing plans.

Grammar:
- A plan is a list of steps separated by commas.
- Within a step, '+' means run primitives in parallel: e.g., "lidar_scan + thermal_snap".
- Allowed primitives ONLY: {allowed}.
- No free text or explanations. Output ONE plan per line or a JSON list of plans.

Examples:
seed: lidar_scan, thermal_snap
rewrites:
- lidar_scan + thermal_snap, gas_sniff
- lidar_scan, gas_sniff, thermal_snap
- thermal_snap + gas_sniff, lidar_scan

Now produce {k} distinct rewrites for the seed.
"""


def build_prompt(seed_plan: List[List[str]], k: int) -> str:
    # turn seed into DSL: comma-separated steps with '+' for parallel
    def fmt_step(step: List[str]) -> str:
        return " + ".join(step)
    seed_text = ", ".join(fmt_step(s) for s in seed_plan)
    return FEW_SHOT.format(allowed=", ".join(PRIMITIVES), k=k) + f"\nseed: {seed_text}\nrewrites:\n"


# -------------------------------
# Seeding
# -------------------------------

def sample_seed(rng: random.Random, *, max_len: int = 4, p_parallel: float = 0.35) -> List[List[str]]:
    """
    Create a small seed plan to bias rewrites.
    """
    L = rng.randint(2, max_len)
    steps: List[List[str]] = []
    pool = PRIMITIVES[:]
    for _ in range(L):
        if rng.random() < p_parallel and len(pool) >= 2:
            a, b = rng.sample(pool, 2)
            steps.append([a, b])
        else:
            a = rng.choice(pool)
            steps.append([a])
    return steps


# -------------------------------
# IO helpers
# -------------------------------

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


# -------------------------------
# generate subcommand
# -------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = raw_dir / f"rewrites_{ts}.jsonl"

    total = args.n
    batch = args.batch_size
    k_per_prompt = args.k_per_prompt

    print(f"[gen] model={args.model} n={total} batch={batch} k_per_prompt={k_per_prompt} T={args.temperature} seed={args.seed}")
    rows: List[Dict[str, Any]] = []

    count = 0
    while count < total:
        # Make one prompt that asks for k_per_prompt rewrites
        seed_plan = sample_seed(rng, max_len=args.seed_max_len, p_parallel=args.seed_p_parallel)
        prompt = build_prompt(seed_plan, k=k_per_prompt)

        try:
            text = call_model(
                args.model,
                prompt,
                temperature=args.temperature,
                seed=args.seed,
                host=args.ollama_host,
                prefer_http=not args.force_cli,
            )
        except Exception as e:
            print(f"[gen][error] model call failed: {e}")
            continue

        # Split into lines first; also try JSON list extraction
        collected: List[str] = []
        # 1) JSON list of strings:
        try:
            blob = re.search(r"(\[[\s\S]+])", text)
            if blob:
                arr = json.loads(blob.group(1))
                if isinstance(arr, list):
                    collected.extend([str(x) for x in arr])
        except Exception:
            pass

        # 2) Linewise
        for ln in text.splitlines():
            ln = ln.strip()
            # remove bullets or numbering
            ln = re.sub(r"^[\-\*\d\.\)\s]+", "", ln)
            if not ln:
                continue
            collected.append(ln)

        # Keep up to batch elements from this call
        for c in collected[:batch]:
            rows.append(
                dict(
                    ts=ts,
                    model=args.model,
                    temperature=args.temperature,
                    seed=args.seed,
                    seed_plan=seed_plan,
                    raw=c,
                    full_response=text[:2000],  # cap for debug
                )
            )
            count += 1
            if count >= total:
                break

        # Flush periodically
        if len(rows) >= 64 or count >= total:
            write_jsonl(out_path, rows)
            print(f"[gen] wrote {len(rows)} rows → {out_path.name}  (total={count}/{total})")
            rows = []

    print(f"[gen] done. file={out_path}")


# -------------------------------
# filter subcommand
# -------------------------------

FEATURE_ORDER = [
    "comp_num_nodes",
    "comp_num_parallel",
    "comp_depth",
    "mean_watts",
    "est_time_ms",
    "includes_safety",
]

@dataclass
class FilterConfig:
    min_nodes: int
    max_nodes: int


def cmd_filter(args: argparse.Namespace) -> None:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all raw files
    raw_dir = in_dir / "raw"
    files = sorted(raw_dir.glob("rewrites_*.jsonl"))
    if not files:
        print(f"[filter] no raw files in {raw_dir}")
        return

    good: List[Dict[str, Any]] = []
    bad: List[Dict[str, Any]] = []
    cfg = FilterConfig(min_nodes=args.min_nodes, max_nodes=args.max_nodes)

    for fp in files:
        rows = load_jsonl(fp)
        for r in rows:
            raw = r.get("raw", "")
            seed_plan = r.get("seed_plan", [])

            try:
                groups = parse_plan_text(raw)
                ok, reason = validate_plan(groups, min_nodes=cfg.min_nodes, max_nodes=cfg.max_nodes)
                if not ok:
                    bad.append(dict(**r, reason=reason))
                    continue

                feats = compute_features(groups)
                # pack features vector
                vec = np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float32)
                nodes, parallel, depth, num_nodes, num_parallel = groups_to_composite(groups)
                rec = dict(
                    seed_plan=seed_plan,
                    parsed_plan=groups,              # list of steps (parallel via len>1)
                    nodes=nodes,                      # flat execution order
                    parallel=parallel,                # {gid: [node_indices]}
                    comp_depth=depth,
                    comp_num_nodes=num_nodes,
                    comp_num_parallel=num_parallel,
                    features=feats,
                    feature_vector=vec.tolist(),
                    raw=raw,
                    model=r.get("model"),
                    temperature=r.get("temperature"),
                    seed=r.get("seed"),
                    ts=r.get("ts"),
                )
                good.append(rec)
            except Exception as e:
                bad.append(dict(**r, reason=f"parse/score error: {e}"))
                print(f"[filter][debug] parse fail for raw='{raw[:120]}' → {e}")

    print(f"[filter] candidates={len(good)+len(bad)}  accepted={len(good)}  rejected={len(bad)}")

    # Deduplicate on plan structure hash
    def plan_key(g: Dict[str, Any]) -> str:
        # canonical key: 'a|b , c , d|e' (parallel joined by '|', steps by ',')
        steps = ["|".join(step) for step in g["parsed_plan"]]
        return ", ".join(steps)

    uniq = {}
    for g in good:
        uniq.setdefault(plan_key(g), g)
    good = list(uniq.values())
    print(f"[filter] unique accepted={len(good)}")

    # Stack features → composites.npy (optionally normalize rows)
    if good:
        M = np.stack([np.array(g["feature_vector"], dtype=np.float32) for g in good], axis=0)
        if args.row_normalize:
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            M = M / norms
        np.save(out_dir / "composites.npy", M.astype(np.float32))
        print(f"[filter] wrote composites.npy with shape {M.shape}")

    # Persist accepted rewrites
    (out_dir / "good_rewrites.json").write_text(json.dumps(good, indent=2))
    print(f"[filter] wrote good_rewrites.json  (records={len(good)})")

    # Optional: save feature schema for clarity
    schema = dict(
        feature_order=FEATURE_ORDER,
        description={
            "comp_num_nodes": "Total primitive count (serial + parallel).",
            "comp_num_parallel": "Number of parallel groups (len(step)>=2).",
            "comp_depth": "1 if serial-only; 2 if any parallel (shallow).",
            "mean_watts": "Mean nominal power across primitives (W).",
            "est_time_ms": "Estimated runtime based on serial-sum / parallel-max heuristic (ms).",
            "includes_safety": "1.0 if any safety-critical primitive is present else 0.0.",
        },
        primitives=PRIMITIVES,
        aliases=ALIASES,
        units={"mean_watts": "W", "est_time_ms": "ms"},
    )
    (out_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2))


# -------------------------------
# CLI
# -------------------------------

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Rewrite pipeline (generate / filter).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    apg = sub.add_parser("generate", help="Sample seeds and call Gemma (Ollama) to rewrite.")
    apg.add_argument("--out_dir", type=Path, required=True, help="Output directory for raw generations.")
    apg.add_argument("--model", type=str, default="gemma3:4b", help="Ollama model name.")
    apg.add_argument("--n", type=int, default=200, help="Total number of candidate lines to collect.")
    apg.add_argument("--batch_size", type=int, default=8, help="Keep up to this many lines per model call.")
    apg.add_argument("--k_per_prompt", type=int, default=8, help="Ask the model to produce k rewrites per seed.")
    apg.add_argument("--temperature", type=float, default=0.7)
    apg.add_argument("--seed", type=int, default=13)
    apg.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="Ollama server URL.")
    apg.add_argument("--force_cli", action="store_true", help="Force CLI fallback instead of HTTP API.")
    apg.add_argument("--seed_max_len", type=int, default=4)
    apg.add_argument("--seed_p_parallel", type=float, default=0.35)

    apf = sub.add_parser("filter", help="Validate + score raw generations to composites.")
    apf.add_argument("--in_dir", type=Path, required=True, help="Input directory containing raw/*.jsonl")
    apf.add_argument("--out_dir", type=Path, required=True, help="Where to write composites.npy and good_rewrites.json")
    apf.add_argument("--max_nodes", type=int, default=8)
    apf.add_argument("--min_nodes", type=int, default=2)
    apf.add_argument("--row_normalize", action="store_true", help="L2-normalize rows in composites.npy")

    return ap


def main():
    ap = build_cli()
    args = ap.parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "filter":
        cmd_filter(args)
    else:
        raise ValueError(f"Unknown cmd={args.cmd}")


if __name__ == "__main__":
    main()

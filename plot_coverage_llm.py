# plot_coverage_llm.py

import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# ---- import your bench definitions ----
from bench import GridSim, SimCfg, FusionHook


# Helper to decide sparingly when to call the LLM
def should_call_llm(pred_ms: np.ndarray, gap_pct_thresh: float, allow_hazard: bool = False,
                    hazard: bool = False) -> bool:
    if not allow_hazard and hazard:
        return False
    if len(pred_ms) < 2:
        return False
    gap_pct = (pred_ms[1] - pred_ms[0]) / max(1e-6, pred_ms[0])
    return gap_pct < gap_pct_thresh


def simulate_trajectory(use_fusion: bool,
                        use_llm: bool = False,
                        llm_model: str = None,
                        top_k: int = 0,
                        llm_gap_pct: float = 0.10):
    cfg = SimCfg(
        size=80,
        obstacle_prob=0.08,
        n_robots=1,
        robot_speed=1.0,
        info_radius=4,
        max_steps=10000,
        coverage_target=0.95,
        seed=0
    )
    sim = GridSim(cfg)
    times = [sim.t]
    covs  = [sim.coverage()]

    fusion = None
    if use_fusion:
        fusion = FusionHook(
            fusion_dir=Path("fusion_runs"),
            imc_vectors=Path("imc_out")/"composites.npy",
            good_json=Path("imc_out")/"good_rewrites.json"
        )

    while sim.coverage() < cfg.coverage_target and sim.t < cfg.max_steps:
        fr = sim.frontiers()
        goal = sim.robots[0] if not fr else min(
            fr, key=lambda f: sim.a_star_dist(sim.robots[0], f) or np.inf
        )

        scan_time = 1.0
        if fusion:
            # build context
            ctx5 = np.array([
                float(cfg.size * cfg.size),
                float(np.mean(sim.gt == 1)),
                0.0, 0.0, 0.0
            ], dtype=np.float32)
            # score all IMC composites
            # returns seconds already (predict_scan_seconds returns ms/1000)
            y_ms = fusion.model_in_ms(ctx5) if hasattr(fusion, 'model_in_ms') else None
            # fallback: predict all then pick first
            if y_ms is None:
                # reuse FusionHook.predict_scan_seconds for best only
                best_idx, _ = fusion.choose_composite(ctx5)
                base_time = fusion.predict_scan_seconds(ctx5)
                y_ms = None  # no full array
            else:
                base_time = y_ms[0] / 1000.0

            # decide whether to call LLM
            hazard = False  # or simulate hazard logic
            if use_llm and llm_model and top_k > 0 and y_ms is not None:
                if should_call_llm(y_ms, llm_gap_pct, hazard=hazard):
                    t0 = time.time()
                    _, scan_time = fusion.choose_composite(
                        ctx5,
                        llm_model=llm_model,
                        top_k=top_k
                    )
                    llm_latency = time.time() - t0
                    sim.t += llm_latency
                else:
                    scan_time = base_time
            else:
                scan_time = base_time

        sim.step_move_and_scan(0, goal, scan_time)
        times.append(sim.t)
        covs.append(sim.coverage())

    return times, covs


def main():
    parser = argparse.ArgumentParser(
        description="Plot coverage vs. time for baseline, fusion-MLP, and fusion+LLM"
    )
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable LLM re-ranking in fusion decisions")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM model name for re-ranking (e.g., gemma3n:e4b)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-K candidates to re-rank with LLM")
    parser.add_argument("--llm-gap-pct", type=float, default=0.10,
                        help="Only call LLM when (top2-top1)/top1 < this threshold")
    args = parser.parse_args()

    t_base, c_base = simulate_trajectory(use_fusion=False)
    t_fus,  c_fus  = simulate_trajectory(
        use_fusion=True,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        top_k=args.top_k,
        llm_gap_pct=args.llm_gap_pct
    )

    plt.figure()
    plt.plot(t_base, c_base, label="Baseline (1 s scan)")
    label = "Fusion-MLP"
    if args.use_llm:
        label += f" + LLM (gap<{args.llm_gap_pct:.2f})"
    plt.plot(t_fus, c_fus, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Coverage")
    plt.title("Coverage vs. Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

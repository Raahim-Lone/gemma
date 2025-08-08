# Graph-Based Cost Learning & Gemma 3n for Rapid Robotic Sensing  
*(Google Gemma 3n Impact Challenge — Hackathon submission)*

We turn an on-device **Gemma 3n** LLM into an action-planning oracle for disaster-response robots.  
The pipeline:

1. **Sim-log** thousands of composite sensing plans in the *LeRobot* simulator  
2. **Export** the resulting time/energy/safety matrix  
3. **Train** an **Inductive Graph-based Matrix Completion (IGMC)** model to predict plan latency in unseen rooms  
4. **Generate & rewrite** new plans with Gemma 3n → embed → k-center sample  
5. **Fuse** IGMC predictions with lightweight contextual features via an MLP  
6. **Deploy**: the runtime loop greedily picks the fastest plan, conditionally re-ranking with Gemma 3n only when the fusion model is uncertain.

<p align="center">
  <img src="docs/fig_pipeline_overview.png" width="680">
</p>

---

## 0. Quick start (TL;DR)

```bash
# ❶ create env (Python ≥3.10)
conda create -n gemma3n_robot python=3.10
conda activate gemma3n_robot
pip install -r requirements.txt           # torch, torch-geometric, kaggle-api, lerobot, ollama-py, ...

# ❷ simulate 20 “rooms” = ~3 min on laptop CPU
python log_sim_le_robot.py --rooms 20 --out lerobot_ds

# ❸ build cost matrix & IGMC dataset
python build_cost_matrix.py  --in_dir lerobot_ds --out_dir matrix_v1
python export_to_igmc.py     --matrix_dir matrix_v1 --out_dir igmc_data

# ❹ train IGMC (GPU optional, ~2 min on RTX 3060)
python train_igmc_robot.py   --data_root igmc_data --epochs 160 --out_dir igmc_runs

# ❺ call Gemma 3n locally via Ollama, filter & embed 2 k rewrites  (⇠5 min)
python rewrite_pipeline.py generate --out_dir imc_out --model gemma3n:e4b --n 2000
python rewrite_pipeline.py filter   --in_dir imc_out --out_dir imc_out --row_normalize

# ❻ sample & execute 120 composites into a dataset
python select_and_execute.py select  --composites imc_out/composites.npy --k 120 --out imc_out/selected_ids.json
python select_and_execute.py execute --good_json imc_out/good_rewrites.json \
       --selected imc_out/selected_ids.json --rooms 40 --dataset_out lerobot_exec_ds

# ❼ train fusion MLP  (~15 s)
python train_fusion_mlp.py --dataset_root lerobot_exec_ds --exec_index lerobot_exec_ds/executed_index.json \
       --imc_vectors imc_out/composites.npy --out_dir fusion_runs

# ❽ runtime demo: 10 new rooms, top-3 plans each, k-center diversity
python runtime_loop.py deploy --fusion_dir fusion_runs --imc_vectors imc_out/composites.npy \
       --good_json imc_out/good_rewrites.json --rooms 10 --per_room 3 \
       --dataset_out runtime_ds --diversity

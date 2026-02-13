# Agent Prompt: Verify Mic-Only Ablation (Does LDV Provide Irreducible Info?)

## Mission
Verify whether LDV provides **unique** predictive signal for band selection, beyond mic-only features, by running a controlled ablation:
- Train/evaluate a DTmin student with **LDV+MIC observations**
- Train/evaluate a DTmin student with **MIC-only observations**

Teacher actions are identical (truth-guided MIC-MIC score); only the student observation differs.

## Inputs (locked)
- Data root: `/home/sbplab/jiawei/data`
- Chirp truth-ref root: `results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000`
- Speakers: `18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V`

## Quickstart (4 commands)

### 1) Teacher trajectories (LDV+MIC obs)
```bash
python -u scripts/teacher_band_omp_micmic.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/ablation_teacher_ldv_mic_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --obs_mode ldv_mic
```

### 2) Teacher trajectories (MIC-only obs)
```bash
python -u scripts/teacher_band_omp_micmic.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/ablation_teacher_mic_only_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --obs_mode mic_only_strict
```

### 3) Train/eval students
```bash
python -u scripts/train_dtmin_from_band_trajectories.py \
  --traj_path results/ablation_teacher_ldv_mic_<ts>/teacher_trajectories.npz \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/ablation_student_ldv_mic_<YYYYMMDD_HHMMSS>
```

```bash
python -u scripts/train_dtmin_from_band_trajectories.py \
  --traj_path results/ablation_teacher_mic_only_<ts>/teacher_trajectories.npz \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/ablation_student_mic_only_<YYYYMMDD_HHMMSS>
```

### 4) Compare
```bash
python -u scripts/compare_ablation_runs.py \
  --run_a results/ablation_student_ldv_mic_<ts> \
  --run_b results/ablation_student_mic_only_<ts> \
  --out_dir results/ablation_compare_<YYYYMMDD_HHMMSS>
```

## Acceptance (Claim 2)
On pooled test windows (vs chirp reference), we accept “LDV provides irreducible info” if:
- p95(theta_error_ref) of LDV+MIC student is ≥ 10% lower than MIC-only student
- fail_rate(theta_error_ref>5°) of LDV+MIC student is ≥ 20% lower than MIC-only student

Read: `results/ablation_compare_<ts>/ablation_report.md`

## Pitfalls
1) Do not change the window split or evaluation; use the plan-locked student script.
2) Do not change teacher scoring; actions must remain MIC-MIC truth-guided.
3) Do not write artifacts to repo root; always under `results/<run_name>/`.


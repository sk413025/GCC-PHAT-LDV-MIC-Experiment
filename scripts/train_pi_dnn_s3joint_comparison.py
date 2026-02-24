"""
PI-DNN vs S3-joint comparison.

Implements the Physics-Informed DNN described in paper/main.tex Sec 3.2
and compares it against the S3-joint signal-processing baseline from
commit 9b680c6 (full_reanalysis_vy025_wider.py).

Usage:
    python scripts/train_pi_dnn_s3joint_comparison.py
"""
import numpy as np
from scipy.io import wavfile
import os
import torch
import torch.nn as nn

# ============================================================
# Constants (copied from full_reanalysis_vy025_wider.py)
# ============================================================
c = 343.0
fs = 48000
d_mic = 1.4
BOARD_Y = 0.25
MIC_Y = 2.0  # microphone distance from wall

# ============================================================
# Dataset definitions: 5 original + 3 re-recordings = 8
# ============================================================
BASE = '/home/sbplab/jiawei/0222-block/0223-block'

datasets = [
    # --- 5 original recordings (from commit 9b680c6) ---
    {'name': 'block-3 #16', 'spk_x': 0.4,
     'path': f'{BASE}/0223-block-3(high)',
     'ml': '0223-MIC-LEFT-40-boy(+0.4m)-16-block.wav',
     'mr': '0223-MIC-RIGHT-40-boy(+0.4m)-16-block.wav',
     'ldv': '0223-LDV-40-boy(+0.4m)-16-block.wav'},
    {'name': 'block-4 #17', 'spk_x': 0.8,
     'path': f'{BASE}/0223-block-4(high)',
     'ml': '0223-MIC-LEFT-40-boy(+0.8m)-17-block.wav',
     'mr': '0223-MIC-RIGHT-40-boy(+0.8m)-17-block.wav',
     'ldv': '0223-LDV-40-boy(+0.8m)-17-block.wav'},
    {'name': 'block-5 #18', 'spk_x': 0.0,
     'path': f'{BASE}/0223-block-5(high)',
     'ml': '0223-MIC-LEFT-40-boy(+0.0m)-18-block.wav',
     'mr': '0223-MIC-RIGHT-40-boy(+0.0m)-18-block.wav',
     'ldv': '0223-LDV-40-boy(+0.0m)-18-block.wav'},
    {'name': 'block-6 #19', 'spk_x': -0.4,
     'path': '/home/sbplab/jiawei/0222-block/0223-block-6(high)',
     'ml': '0223-MIC-LEFT-40-boy(-0.4m)-19-block.wav',
     'mr': '0223-MIC-RIGHT-40-boy(-0.4m)-19-block.wav',
     'ldv': '0223-LDV-40-boy(-0.4m)-19-block.wav'},
    {'name': 'block-7 #20', 'spk_x': -0.8,
     'path': f'{BASE}/0223-block-7(high)',
     'ml': '0223-MIC-LEFT-40-boy(-0.8m)-20-block.wav',
     'mr': '0223-MIC-RIGHT-40-boy(-0.8m)-20-block.wav',
     'ldv': '0223-LDV-40-boy(-0.8m)-20-block.wav'},
    # --- 3 re-recordings ---
    {'name': 'block-4 #21', 'spk_x': 0.8,
     'path': f'{BASE}/0223-block-4(high)',
     'ml': '0223-MIC-LEFT-40-boy(+0.8m)-21-block.wav',
     'mr': '0223-MIC-RIGHT-40-boy(+0.8m)-21-block.wav',
     'ldv': '0223-LDV-40-boy(+0.8m)-21-block.wav'},
    {'name': 'block-5 #22', 'spk_x': 0.0,
     'path': f'{BASE}/0223-block-5(high)',
     'ml': '0223-MIC-LEFT-40-boy(+0.0m)-22-block.wav',
     'mr': '0223-MIC-RIGHT-40-boy(+0.0m)-22-block.wav',
     'ldv': '0223-LDV-40-boy(+0.0m)-22-block.wav'},
    {'name': 'block-7 #21', 'spk_x': -0.8,
     'path': f'{BASE}/0223-block-7(high)',
     'ml': '0223-MIC-LEFT-40-boy(-0.8m)-21-block.wav',
     'mr': '0223-MIC-RIGHT-40-boy(-0.8m)-21-block.wav',
     'ldv': '0223-LDV-40-boy(-0.8m)-21-block.wav'},
]

# ============================================================
# GCC-PHAT (exact copy from full_reanalysis_vy025_wider.py)
# ============================================================
def gcc_phat(sig1, sig2, fs):
    n = len(sig1) + len(sig2) - 1
    nfft = 2**int(np.ceil(np.log2(n)))
    X1 = np.fft.rfft(sig1, n=nfft)
    X2 = np.fft.rfft(sig2, n=nfft)
    G = X1 * np.conj(X2)
    denom = np.abs(G); denom[denom < 1e-15] = 1e-15
    gcc = np.fft.irfft(G / denom, n=nfft)
    gcc = np.concatenate([gcc[nfft//2:], gcc[:nfft//2]])
    lags = np.arange(-nfft//2, nfft//2) / fs
    return gcc, lags


def find_peak_in_range(gcc, lags, center_ms, hw=0.5):
    """Find highest peak within [center_ms - hw, center_ms + hw]."""
    mask = (lags*1000 >= center_ms - hw) & (lags*1000 <= center_ms + hw)
    if not np.any(mask):
        return np.nan, np.nan
    idx = np.where(mask)[0][np.argmax(gcc[mask])]
    return lags[idx]*1000, gcc[idx]


def s3_joint(gcc_vl, lags_vl, gcc_vr, lags_vr, spk_x, hw=0.5):
    """S3-joint from commit 9b680c6: 1D sweep with Δτ locked to theory.

    1. Compute theoretical τ_VL, τ_VR from v=(spk_x, BOARD_Y)
    2. Scan τ_VL candidates within ±hw ms of τ_VL_theory
    3. For each τ_VL: fix τ_VR = τ_VL + Δτ_theory
    4. Joint score = cc_vl(τ_VL) + cc_vr(τ_VR)
    5. Pick τ_VL with highest score → Δτ = τ_VR - τ_VL ≡ Δτ_theory
    """
    my = MIC_Y - BOARD_Y  # vertical distance from vibration point to mic plane
    d_vl = np.sqrt((spk_x + 0.7)**2 + my**2)
    d_vr = np.sqrt((spk_x - 0.7)**2 + my**2)
    tau_vl_theory_ms = -d_vl / c * 1000  # negative lag
    tau_vr_theory_ms = -d_vr / c * 1000
    delta_tau_theory_ms = tau_vr_theory_ms - tau_vl_theory_ms

    # Scan τ_VL around theoretical value
    vl_mask = ((lags_vl * 1000 >= tau_vl_theory_ms - hw) &
               (lags_vl * 1000 <= tau_vl_theory_ms + hw))
    if not np.any(vl_mask):
        return None

    vl_indices = np.where(vl_mask)[0]
    best_score = -np.inf
    best_tvl = None
    best_tvr = None

    for vl_idx in vl_indices:
        tvl_ms = lags_vl[vl_idx] * 1000
        avl = gcc_vl[vl_idx]

        # τ_VR is locked: τ_VR = τ_VL + Δτ_theory
        tvr_target_ms = tvl_ms + delta_tau_theory_ms
        # Find nearest lag index for τ_VR
        vr_idx = np.argmin(np.abs(lags_vr * 1000 - tvr_target_ms))
        avr = gcc_vr[vr_idx]

        score = avl + avr
        if score > best_score:
            best_score = score
            best_tvl = tvl_ms
            best_tvr = lags_vr[vr_idx] * 1000

    if best_tvl is None:
        return None

    dt = best_tvr - best_tvl  # ≈ delta_tau_theory_ms
    return (best_tvl, best_tvr, dt, best_score)


# ============================================================
# Step 7: S3-joint Baseline (full-length WAV, no windowing)
# ============================================================
def run_s3joint_baseline():
    """Reproduce S3-joint results from commit 9b680c6 using only the
    5 original recordings."""
    print("=" * 90)
    print("S3-joint Baseline Verification (commit 9b680c6)")
    print("=" * 90)

    original_ds = datasets[:5]  # only the 5 originals
    results = {}

    for ds in original_ds:
        spk_x = ds['spk_x']
        # True DoA from speaker-mic geometry
        d_SL = np.sqrt((spk_x + 0.7)**2 + MIC_Y**2)
        d_SR = np.sqrt((spk_x - 0.7)**2 + MIC_Y**2)
        tau_mic_ms = (d_SL - d_SR) / c * 1000
        theta_true = np.degrees(np.arcsin(np.clip(tau_mic_ms / 1000 * c / d_mic, -1, 1)))

        # Load WAV
        _, ml = wavfile.read(os.path.join(ds['path'], ds['ml']))
        _, mr = wavfile.read(os.path.join(ds['path'], ds['mr']))
        _, ldv = wavfile.read(os.path.join(ds['path'], ds['ldv']))
        ml = ml.astype(np.float64); mr = mr.astype(np.float64); ldv = ldv.astype(np.float64)

        # GCC-PHAT
        gcc_vl, lags_vl = gcc_phat(ldv, ml, fs)
        gcc_vr, lags_vr = gcc_phat(ldv, mr, fs)

        s3 = s3_joint(gcc_vl, lags_vl, gcc_vr, lags_vr, spk_x, hw=0.5)
        if s3:
            _, _, dt, _ = s3
            theta_s3 = np.degrees(np.arcsin(np.clip(dt / 1000 * c / d_mic, -1, 1)))
            err = theta_s3 - theta_true
        else:
            theta_s3 = float('nan'); err = float('nan')

        results[spk_x] = {'theta_true': theta_true, 'theta_s3': theta_s3, 'err': err}
        print(f"  spk_x={spk_x:+.1f}m  theta_true={theta_true:+.2f}°  "
              f"S3-joint={theta_s3:+.2f}°  err={err:+.2f}°")

    mae = np.mean([abs(r['err']) for r in results.values()])
    print(f"\n  S3-joint MAE = {mae:.2f}°")

    # Sanity check: expected results from commit 9b680c6
    expected = {0.4: 1.37, 0.8: 2.33, 0.0: 0.00, -0.4: -1.37, -0.8: -2.33}
    ok = True
    for spk_x, exp_err in expected.items():
        actual = results[spk_x]['err']
        if abs(actual - exp_err) > 0.1:
            print(f"  *** MISMATCH: spk_x={spk_x:+.1f} expected err={exp_err:+.2f} got {actual:+.2f}")
            ok = False
    if ok:
        print("  [OK] S3-joint results match commit 9b680c6")
    else:
        print("  [WARN] S3-joint results differ from commit 9b680c6!")

    return results, mae


# ============================================================
# Step 3: GCC-PHAT Feature Extraction (windowed)
# ============================================================
# Lag crop range: [-8.5ms, -3.5ms]
LAG_START_MS = -8.5
LAG_END_MS = -3.5
WIN_SEC = 0.5
STRIDE_SEC = 0.25
WIN_SAMPLES = int(WIN_SEC * fs)
STRIDE_SAMPLES = int(STRIDE_SEC * fs)


def extract_windowed_features():
    """Extract windowed GCC-PHAT features from all 8 recordings."""
    all_features = []   # (N, 480)
    all_gcc_vl = []     # (N, 240) — raw GCC curves for physics loss
    all_gcc_vr = []     # (N, 240)
    all_theta_true = [] # (N,) — true DoA angle
    all_spk_x = []      # (N,) — speaker x for position label
    lag_ms_axis = None   # shared lag axis (240,)

    for ds in datasets:
        spk_x = ds['spk_x']
        d_SL = np.sqrt((spk_x + 0.7)**2 + MIC_Y**2)
        d_SR = np.sqrt((spk_x - 0.7)**2 + MIC_Y**2)
        tau_mic_ms = (d_SL - d_SR) / c * 1000
        theta_true = np.degrees(np.arcsin(np.clip(tau_mic_ms / 1000 * c / d_mic, -1, 1)))

        _, ml = wavfile.read(os.path.join(ds['path'], ds['ml']))
        _, mr = wavfile.read(os.path.join(ds['path'], ds['mr']))
        _, ldv = wavfile.read(os.path.join(ds['path'], ds['ldv']))
        ml = ml.astype(np.float64); mr = mr.astype(np.float64); ldv = ldv.astype(np.float64)

        n_samples = min(len(ml), len(mr), len(ldv))
        n_windows = (n_samples - WIN_SAMPLES) // STRIDE_SAMPLES + 1

        for w in range(n_windows):
            start = w * STRIDE_SAMPLES
            end = start + WIN_SAMPLES
            ldv_win = ldv[start:end]
            ml_win = ml[start:end]
            mr_win = mr[start:end]

            gvl, lags = gcc_phat(ldv_win, ml_win, fs)
            gvr, _    = gcc_phat(ldv_win, mr_win, fs)
            lags_ms = lags * 1000

            # Crop to [-8.5, -3.5] ms
            mask = (lags_ms >= LAG_START_MS) & (lags_ms <= LAG_END_MS)
            gvl_crop = gvl[mask]
            gvr_crop = gvr[mask]

            if lag_ms_axis is None:
                lag_ms_axis = lags_ms[mask]

            # Ensure consistent length (should be ~240 points)
            n_pts = len(lag_ms_axis)
            gvl_crop = gvl_crop[:n_pts]
            gvr_crop = gvr_crop[:n_pts]

            feature = np.concatenate([gvl_crop, gvr_crop])
            all_features.append(feature)
            all_gcc_vl.append(gvl_crop)
            all_gcc_vr.append(gvr_crop)
            all_theta_true.append(theta_true)
            all_spk_x.append(spk_x)

    all_features = np.array(all_features, dtype=np.float32)
    all_gcc_vl = np.array(all_gcc_vl, dtype=np.float32)
    all_gcc_vr = np.array(all_gcc_vr, dtype=np.float32)
    all_theta_true = np.array(all_theta_true, dtype=np.float32)
    all_spk_x = np.array(all_spk_x, dtype=np.float32)

    print(f"\nFeature extraction: {len(all_features)} windows, "
          f"input_dim={all_features.shape[1]}, "
          f"lag_points={len(lag_ms_axis)}")

    return (all_features, all_gcc_vl, all_gcc_vr,
            all_theta_true, all_spk_x, lag_ms_axis)


# ============================================================
# Step 4: PI-DNN Model (Paper Sec 3.2)
# ============================================================
class PIDNN(nn.Module):
    """Physics-Informed DNN: 480 → 128 → 64 → 2 (x, y)."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        # Initialize output bias to feasible region center
        with torch.no_grad():
            self.net[-1].bias[0] = 0.0   # x ~ 0
            self.net[-1].bias[1] = 0.25  # y ~ BOARD_Y

    def forward(self, x):
        return self.net(x)  # (batch, 2): (x_hat, y_hat)


# ============================================================
# Step 5: Loss Functions
# ============================================================
def compute_theta_from_pos(p_hat):
    """Compute DoA angle (degrees) from predicted 2D position.
    θ = arcsin(Δτ_mic * c / d_mic), where Δτ_mic = (d_SL - d_SR) / c.
    d_SL = dist(speaker, mic_L), d_SR = dist(speaker, mic_R).
    But speaker ≈ p_hat since we're predicting the source position.
    mic_L @ (-0.7, 2.0), mic_R @ (+0.7, 2.0).
    """
    x_hat = p_hat[:, 0]
    y_hat = p_hat[:, 1]
    d_SL = torch.sqrt((x_hat + 0.7)**2 + (MIC_Y - y_hat)**2 + 1e-8)
    d_SR = torch.sqrt((x_hat - 0.7)**2 + (MIC_Y - y_hat)**2 + 1e-8)
    tau_diff = (d_SL - d_SR) / c
    sin_theta = torch.clamp(tau_diff * c / d_mic, -0.999, 0.999)
    theta_deg = torch.rad2deg(torch.asin(sin_theta))
    return theta_deg


def compute_tau_vm(p_hat):
    """Compute theoretical LDV-Mic delays (ms) from predicted position.
    τ_VL = -dist(p_hat, mic_L) / c * 1000   (negative: LDV leads)
    τ_VR = -dist(p_hat, mic_R) / c * 1000
    mic_L @ (-0.7, 2.0), mic_R @ (+0.7, 2.0)
    """
    x_hat = p_hat[:, 0]
    y_hat = p_hat[:, 1]
    d_vl = torch.sqrt((x_hat + 0.7)**2 + (MIC_Y - y_hat)**2 + 1e-8)
    d_vr = torch.sqrt((x_hat - 0.7)**2 + (MIC_Y - y_hat)**2 + 1e-8)
    tau_vl_ms = -d_vl / c * 1000.0
    tau_vr_ms = -d_vr / c * 1000.0
    return tau_vl_ms, tau_vr_ms


def differentiable_interp(gcc_curve, tau_ms, lag_start_ms, lag_step_ms, n_lags):
    """Differentiable linear interpolation into GCC-PHAT curve.
    gcc_curve: (batch, n_lags)
    tau_ms: (batch,) — query delay in ms
    Returns: (batch,) — interpolated GCC value
    """
    idx_float = (tau_ms - lag_start_ms) / lag_step_ms
    idx_float = torch.clamp(idx_float, 0.0, n_lags - 1.001)
    idx_lo = idx_float.long()
    idx_hi = idx_lo + 1
    idx_hi = torch.clamp(idx_hi, max=n_lags - 1)
    frac = idx_float - idx_lo.float()

    # Gather values
    val_lo = gcc_curve.gather(1, idx_lo.unsqueeze(1)).squeeze(1)
    val_hi = gcc_curve.gather(1, idx_hi.unsqueeze(1)).squeeze(1)

    return val_lo * (1.0 - frac) + val_hi * frac


# ============================================================
# Step 6: Training
# ============================================================
def train_pi_dnn(features, gcc_vl, gcc_vr, theta_true, spk_x,
                 lag_ms_axis, lam, seed=42):
    """Train PI-DNN with given lambda. Returns per-angle median predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = features.shape[1]
    n_lags = gcc_vl.shape[1]
    lag_step_ms = float(lag_ms_axis[1] - lag_ms_axis[0])
    lag_start_ms_val = float(lag_ms_axis[0])

    # Tensors
    X = torch.tensor(features)
    gcc_vl_t = torch.tensor(gcc_vl)
    gcc_vr_t = torch.tensor(gcc_vr)
    theta_t = torch.tensor(theta_true)
    spk_x_t = torch.tensor(spk_x)

    model = PIDNN(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_samples = len(X)
    batch_size = 32
    warmup_epochs = 50
    total_epochs = 200

    for epoch in range(total_epochs):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_phys = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb = X[idx]
            gvl_b = gcc_vl_t[idx]
            gvr_b = gcc_vr_t[idx]
            theta_b = theta_t[idx]

            p_hat = model(xb)
            theta_pred = compute_theta_from_pos(p_hat)

            # L_MSE
            loss_mse = torch.mean((theta_pred - theta_b)**2)

            # L_physics (only after warmup)
            if epoch >= warmup_epochs and lam > 0:
                tau_vl_ms, tau_vr_ms = compute_tau_vm(p_hat)
                r_vl = differentiable_interp(gvl_b, tau_vl_ms,
                                             lag_start_ms_val, lag_step_ms, n_lags)
                r_vr = differentiable_interp(gvr_b, tau_vr_ms,
                                             lag_start_ms_val, lag_step_ms, n_lags)
                loss_phys = torch.mean(r_vl + r_vr)
            else:
                loss_phys = torch.tensor(0.0)

            # L_PI = L_MSE - lambda * L_physics
            loss = loss_mse - lam * loss_phys if epoch >= warmup_epochs else loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_phys += loss_phys.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_l = epoch_loss / n_batches
            avg_m = epoch_mse / n_batches
            avg_p = epoch_phys / n_batches
            print(f"    Epoch {epoch+1:3d}  loss={avg_l:+.4f}  "
                  f"MSE={avg_m:.4f}  physics={avg_p:.4f}")

    # --- Evaluation: per-angle median ---
    model.eval()
    with torch.no_grad():
        p_all = model(X)
        theta_all = compute_theta_from_pos(p_all).numpy()
        p_all_np = p_all.numpy()

    unique_angles = sorted(set(theta_true.tolist()))
    angle_results = {}
    for theta in unique_angles:
        mask = np.isclose(theta_true, theta, atol=0.01)
        preds = theta_all[mask]
        positions = p_all_np[mask]
        median_theta = float(np.median(preds))
        spk_x_val = float(spk_x[mask][0])
        mean_pos = positions.mean(axis=0)
        angle_results[theta] = {
            'median_theta': median_theta,
            'spk_x': spk_x_val,
            'mean_xy': (float(mean_pos[0]), float(mean_pos[1])),
            'std_theta': float(np.std(preds)),
        }

    return angle_results


# ============================================================
# Step 8: Leave-One-Angle-Out Cross-Validation
# ============================================================
def train_pi_dnn_cv(features, gcc_vl, gcc_vr, theta_true, spk_x,
                    lag_ms_axis, lam, held_out_theta, seed=42):
    """Train on all angles EXCEPT held_out_theta, evaluate on held_out_theta."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = features.shape[1]
    n_lags = gcc_vl.shape[1]
    lag_step_ms = float(lag_ms_axis[1] - lag_ms_axis[0])
    lag_start_ms_val = float(lag_ms_axis[0])

    # Split train/test by angle
    train_mask = ~np.isclose(theta_true, held_out_theta, atol=0.01)
    test_mask = np.isclose(theta_true, held_out_theta, atol=0.01)

    X_train = torch.tensor(features[train_mask])
    gvl_train = torch.tensor(gcc_vl[train_mask])
    gvr_train = torch.tensor(gcc_vr[train_mask])
    theta_train = torch.tensor(theta_true[train_mask])

    X_test = torch.tensor(features[test_mask])
    gvl_test = torch.tensor(gcc_vl[test_mask])
    gvr_test = torch.tensor(gcc_vr[test_mask])

    model = PIDNN(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_train = len(X_train)
    batch_size = 32
    warmup_epochs = 50
    total_epochs = 200

    for epoch in range(total_epochs):
        model.train()
        perm = torch.randperm(n_train)

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx]
            gvl_b = gvl_train[idx]
            gvr_b = gvr_train[idx]
            theta_b = theta_train[idx]

            p_hat = model(xb)
            theta_pred = compute_theta_from_pos(p_hat)
            loss_mse = torch.mean((theta_pred - theta_b)**2)

            if epoch >= warmup_epochs and lam > 0:
                tau_vl_ms, tau_vr_ms = compute_tau_vm(p_hat)
                r_vl = differentiable_interp(gvl_b, tau_vl_ms,
                                             lag_start_ms_val, lag_step_ms, n_lags)
                r_vr = differentiable_interp(gvr_b, tau_vr_ms,
                                             lag_start_ms_val, lag_step_ms, n_lags)
                loss_phys = torch.mean(r_vl + r_vr)
            else:
                loss_phys = torch.tensor(0.0)

            loss = loss_mse - lam * loss_phys if epoch >= warmup_epochs else loss_mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on held-out angle
    model.eval()
    with torch.no_grad():
        p_test = model(X_test)
        theta_preds = compute_theta_from_pos(p_test).numpy()
        p_test_np = p_test.numpy()

    median_theta = float(np.median(theta_preds))
    mean_pos = p_test_np.mean(axis=0)
    return {
        'median_theta': median_theta,
        'mean_xy': (float(mean_pos[0]), float(mean_pos[1])),
        'std_theta': float(np.std(theta_preds)),
        'n_test': int(test_mask.sum()),
        'n_train': int(train_mask.sum()),
    }


# ============================================================
# Step 9: Main — run everything
# ============================================================
def main():
    # --- S3-joint baseline ---
    s3_results, s3_mae = run_s3joint_baseline()

    # --- Extract features ---
    print("\n" + "=" * 90)
    print("GCC-PHAT Windowed Feature Extraction")
    print("=" * 90)
    (features, gcc_vl, gcc_vr,
     theta_true, spk_x, lag_ms_axis) = extract_windowed_features()

    # ================================================================
    # Part A: Train=Test (original experiment, for reference)
    # ================================================================
    lambdas = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
    dnn_results = {}

    for lam in lambdas:
        print(f"\n{'=' * 90}")
        print(f"[Train=Test] Training PI-DNN with lambda={lam}")
        print(f"{'=' * 90}")
        angle_results = train_pi_dnn(features, gcc_vl, gcc_vr,
                                      theta_true, spk_x, lag_ms_axis,
                                      lam=lam)
        dnn_results[lam] = angle_results

        print(f"\n    Per-angle results (lambda={lam}):")
        for theta, res in sorted(angle_results.items()):
            err = res['median_theta'] - theta
            xy = res['mean_xy']
            print(f"      theta_true={theta:+.2f}°  pred={res['median_theta']:+.2f}°  "
                  f"err={err:+.2f}°  pos=({xy[0]:+.3f},{xy[1]:.3f})  "
                  f"true_pos=({res['spk_x']:+.1f},{BOARD_Y})")

    # ================================================================
    # Part B: Leave-One-Angle-Out Cross-Validation
    # ================================================================
    unique_thetas = sorted(set(theta_true.tolist()))
    # Map theta -> spk_x for display
    theta_to_spkx = {}
    for theta in unique_thetas:
        mask = np.isclose(theta_true, theta, atol=0.01)
        theta_to_spkx[theta] = float(spk_x[mask][0])

    cv_results = {}  # {lam: {theta: result_dict}}

    for lam in lambdas:
        print(f"\n{'=' * 90}")
        print(f"[LOO-CV] Leave-One-Angle-Out with lambda={lam}")
        print(f"{'=' * 90}")
        cv_results[lam] = {}

        for held_out_theta in unique_thetas:
            spk_x_val = theta_to_spkx[held_out_theta]
            res = train_pi_dnn_cv(features, gcc_vl, gcc_vr,
                                   theta_true, spk_x, lag_ms_axis,
                                   lam=lam, held_out_theta=held_out_theta)
            cv_results[lam][held_out_theta] = res
            err = res['median_theta'] - held_out_theta
            print(f"    held_out={held_out_theta:+.2f}° (spk_x={spk_x_val:+.1f}m)  "
                  f"pred={res['median_theta']:+.2f}°  err={err:+.2f}°  "
                  f"train={res['n_train']} test={res['n_test']}")

    # ================================================================
    # Combined Comparison Table
    # ================================================================
    positions = sorted(s3_results.keys())
    thetas_for_pos = {}
    for spk_x_val in positions:
        thetas_for_pos[spk_x_val] = s3_results[spk_x_val]['theta_true']

    # --- Table 1: Train=Test ---
    print(f"\n\n{'=' * 110}")
    print("TABLE 1: Train=Test (per-angle error in degrees) — OVERFITTING REFERENCE")
    print(f"{'=' * 110}")

    lam_strs = [f"  lam={l}" for l in lambdas]
    hdr = f"{'Position':>10} {'theta_true':>10} {'S3-joint':>10}"
    for ls in lam_strs:
        hdr += f"{ls:>10}"
    print(hdr)
    print("-" * len(hdr))

    all_s3_errs = []
    all_dnn_errs_tt = {l: [] for l in lambdas}

    for spk_x_val in positions:
        theta_true_val = thetas_for_pos[spk_x_val]
        s3_err = s3_results[spk_x_val]['err']
        all_s3_errs.append(abs(s3_err))

        row = f"{spk_x_val:+.1f}m".rjust(10)
        row += f"{theta_true_val:+.2f}".rjust(10) + "°"
        row += f"{s3_err:+.2f}".rjust(9) + "°"

        for lam in lambdas:
            ar = dnn_results[lam]
            match = None
            for theta, res in ar.items():
                if abs(theta - theta_true_val) < 0.1:
                    match = res; break
            if match:
                dnn_err = match['median_theta'] - theta_true_val
                all_dnn_errs_tt[lam].append(abs(dnn_err))
                row += f"{dnn_err:+.2f}".rjust(9) + "°"
            else:
                row += "N/A".rjust(10)
        print(row)

    s3_mae_val = np.mean(all_s3_errs)
    row_mae = "MAE".rjust(10) + " ".rjust(10) + " "
    row_mae += f"{s3_mae_val:.2f}".rjust(9) + "°"
    for lam in lambdas:
        if all_dnn_errs_tt[lam]:
            row_mae += f"{np.mean(all_dnn_errs_tt[lam]):.2f}".rjust(9) + "°"
        else:
            row_mae += "N/A".rjust(10)
    print("-" * len(hdr))
    print(row_mae)

    # --- Table 2: LOO-CV ---
    print(f"\n\n{'=' * 110}")
    print("TABLE 2: Leave-One-Angle-Out CV (per-angle error in degrees) — GENERALIZATION TEST")
    print(f"{'=' * 110}")
    print(hdr)
    print("-" * len(hdr))

    all_dnn_errs_cv = {l: [] for l in lambdas}

    for spk_x_val in positions:
        theta_true_val = thetas_for_pos[spk_x_val]
        s3_err = s3_results[spk_x_val]['err']

        row = f"{spk_x_val:+.1f}m".rjust(10)
        row += f"{theta_true_val:+.2f}".rjust(10) + "°"
        row += f"{s3_err:+.2f}".rjust(9) + "°"

        for lam in lambdas:
            # Find the CV result for this theta
            match = None
            for theta, res in cv_results[lam].items():
                if abs(theta - theta_true_val) < 0.1:
                    match = res; break
            if match:
                dnn_err = match['median_theta'] - theta_true_val
                all_dnn_errs_cv[lam].append(abs(dnn_err))
                row += f"{dnn_err:+.2f}".rjust(9) + "°"
            else:
                row += "N/A".rjust(10)
        print(row)

    row_mae = "MAE".rjust(10) + " ".rjust(10) + " "
    row_mae += f"{s3_mae_val:.2f}".rjust(9) + "°"
    for lam in lambdas:
        if all_dnn_errs_cv[lam]:
            row_mae += f"{np.mean(all_dnn_errs_cv[lam]):.2f}".rjust(9) + "°"
        else:
            row_mae += "N/A".rjust(10)
    print("-" * len(hdr))
    print(row_mae)

    # --- Summary ---
    print(f"\n\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(f"  S3-joint MAE = {s3_mae_val:.2f}°")
    for lam in lambdas:
        tt_mae = np.mean(all_dnn_errs_tt[lam]) if all_dnn_errs_tt[lam] else float('nan')
        cv_mae = np.mean(all_dnn_errs_cv[lam]) if all_dnn_errs_cv[lam] else float('nan')
        gap = cv_mae - tt_mae
        flag = "  <<<OVERFIT" if gap > 1.0 else ""
        print(f"  lambda={lam:<5}  Train=Test MAE={tt_mae:.2f}°  "
              f"LOO-CV MAE={cv_mae:.2f}°  gap={gap:+.2f}°{flag}")

    best_cv_lam = None
    best_cv_mae = 999.0
    for lam in lambdas:
        if all_dnn_errs_cv[lam]:
            cv_mae = np.mean(all_dnn_errs_cv[lam])
            if cv_mae < best_cv_mae:
                best_cv_mae = cv_mae
                best_cv_lam = lam

    print(f"\n  Best CV lambda = {best_cv_lam}  (LOO-CV MAE = {best_cv_mae:.2f}°)")
    if best_cv_mae <= s3_mae_val:
        print(f"  [OK] PI-DNN LOO-CV MAE={best_cv_mae:.2f}° <= S3-joint MAE={s3_mae_val:.2f}°")
    else:
        print(f"  [WARN] PI-DNN LOO-CV MAE={best_cv_mae:.2f}° > S3-joint MAE={s3_mae_val:.2f}°")
        print(f"  DNN cannot generalize to unseen angles better than S3-joint.")


if __name__ == '__main__':
    main()

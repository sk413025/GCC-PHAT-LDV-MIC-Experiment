#!/usr/bin/env python3
"""
Compute COMPLETE chirp Table 1 data with two search windows (+-2.5ms and +-4.08ms).
Covers all 5 positions, unblock MIC-MIC, block MIC-MIC, block S3-joint.
Also processes alternate recordings for +-0.8m positions.
"""

import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
import os

# ============================================================
# Constants
# ============================================================
c = 343.0
fs_expected = 48000
MIC_L = np.array([-0.7, 2.0])
MIC_R = np.array([+0.7, 2.0])
d_mic = 1.4
BOARD_Y = 0.25

# ============================================================
# Core functions
# ============================================================
def gcc_phat(sig1, sig2, fs, max_tau_ms):
    n = len(sig1)
    nfft = 2**int(np.ceil(np.log2(n)))
    X1 = fft(sig1.astype(np.float64), nfft)
    X2 = fft(sig2.astype(np.float64), nfft)
    G = X1 * np.conj(X2)
    denom = np.abs(G)
    denom[denom < 1e-10] = 1e-10
    gcc = np.real(ifft(G / denom))

    max_samples = int(max_tau_ms / 1000.0 * fs)
    gcc_trimmed = np.concatenate([gcc[-max_samples:], gcc[:max_samples+1]])
    lags = np.arange(-max_samples, max_samples+1)

    peak_idx = np.argmax(gcc_trimmed)
    peak_lag = lags[peak_idx]
    tau_ms = peak_lag / fs * 1000.0

    # Top 5 peaks
    peaks_idx, _ = find_peaks(gcc_trimmed)
    if len(peaks_idx) == 0:
        peaks_idx = np.array([peak_idx])
    peak_vals = gcc_trimmed[peaks_idx]
    sorted_idx = np.argsort(peak_vals)[::-1][:5]
    top5 = [(lags[peaks_idx[i]] / fs * 1000.0, gcc_trimmed[peaks_idx[i]]) for i in sorted_idx]

    return tau_ms, gcc_trimmed[peak_idx], gcc_trimmed, lags, top5


def tau_to_doa(tau_ms, c=343.0, d_mic=1.4):
    tau_s = tau_ms / 1000.0
    sin_val = tau_s * c / d_mic
    sin_val = np.clip(sin_val, -1, 1)
    return np.degrees(np.arcsin(sin_val))


def compute_true_angle(spk_x):
    """Compute true DoA angle from geometry."""
    d_L = np.sqrt((spk_x - (-0.7))**2 + 2.0**2)
    d_R = np.sqrt((spk_x - 0.7)**2 + 2.0**2)
    tau_true = (d_L - d_R) / c
    sin_val = tau_true * c / d_mic
    sin_val = np.clip(sin_val, -1, 1)
    return np.degrees(np.arcsin(sin_val)), tau_true * 1000.0


def s3_joint(ldv, mic_l, mic_r, fs, spk_x, board_y=0.25, W_ms=0.5):
    """
    S3-joint: constrained joint search using geometric prior.
    """
    v = np.array([spk_x, board_y])
    mic_l_pos = np.array([-0.7, 2.0])
    mic_r_pos = np.array([+0.7, 2.0])

    d_vl = np.linalg.norm(v - mic_l_pos)
    d_vr = np.linalg.norm(v - mic_r_pos)

    tau_vl_theory = -d_vl / c
    tau_vr_theory = -d_vr / c
    delta_tau_theory = tau_vr_theory - tau_vl_theory  # (d_vl - d_vr) / c

    n = len(ldv)
    nfft = 2**int(np.ceil(np.log2(n)))

    def compute_gcc_full(sig1, sig2):
        X1 = fft(sig1.astype(np.float64), nfft)
        X2 = fft(sig2.astype(np.float64), nfft)
        G = X1 * np.conj(X2)
        denom = np.abs(G)
        denom[denom < 1e-10] = 1e-10
        return np.real(ifft(G / denom))

    gcc_vl = compute_gcc_full(ldv, mic_l)
    gcc_vr = compute_gcc_full(ldv, mic_r)

    W_samples = int(W_ms / 1000.0 * fs)
    center_sample = int(round(tau_vl_theory * fs))
    delta_samples = int(round(delta_tau_theory * fs))

    best_score = -np.inf
    best_tau_vl = None

    for offset in range(-W_samples, W_samples + 1):
        idx_vl = (center_sample + offset) % nfft
        idx_vr = (center_sample + offset + delta_samples) % nfft
        score = gcc_vl[idx_vl] + gcc_vr[idx_vr]
        if score > best_score:
            best_score = score
            best_tau_vl = (center_sample + offset) / fs

    best_tau_vr = best_tau_vl + delta_tau_theory
    delta_tau = best_tau_vr - best_tau_vl  # equals delta_tau_theory

    doa = tau_to_doa(delta_tau * 1000.0)

    return doa, delta_tau * 1000.0, tau_vl_theory * 1000.0, tau_vr_theory * 1000.0


def load_wav(path):
    """Load wav file, return mono float64 signal and sample rate."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    return sr, data.astype(np.float64)


# ============================================================
# File definitions
# ============================================================
base = "/home/sbplab/jiawei/0222-block"

# Block files: (position_label, spk_x, directory, recording_numbers, suffix)
block_files = {
    "+0.0m": {
        "spk_x": 0.0,
        "dir": f"{base}/0223-block/0223-block-5(high)",
        "recordings": {
            18: "primary",
            22: "alternate",
        },
    },
    "+0.4m": {
        "spk_x": 0.4,
        "dir": f"{base}/0223-block/0223-block-3(high)",
        "recordings": {
            16: "primary",
        },
    },
    "+0.8m": {
        "spk_x": 0.8,
        "dir": f"{base}/0223-block/0223-block-4(high)",
        "recordings": {
            17: "primary",
            21: "alternate",
        },
    },
    "-0.4m": {
        "spk_x": -0.4,
        "dir": f"{base}/0223-block-6(high)",
        "recordings": {
            19: "primary",
        },
    },
    "-0.8m": {
        "spk_x": -0.8,
        "dir": f"{base}/0223-block/0223-block-7(high)",
        "recordings": {
            20: "primary",
            21: "alternate",
        },
    },
}

unblock_files = {
    "+0.0m": {
        "spk_x": 0.0,
        "dir": f"{base}/0223-unblock-5(high)",
        "recordings": {18: "primary"},
    },
    "+0.4m": {
        "spk_x": 0.4,
        "dir": f"{base}/0223-unblock-3(high)",
        "recordings": {14: "primary"},
    },
    "+0.8m": {
        "spk_x": 0.8,
        "dir": f"{base}/0223-unblock-4(high)",
        "recordings": {17: "primary"},
    },
    "-0.4m": {
        "spk_x": -0.4,
        "dir": f"{base}/0223-unblock-6(high)",
        "recordings": {19: "primary"},
    },
    "-0.8m": {
        "spk_x": -0.8,
        "dir": f"{base}/0223-block/0223-unblock-7(high)",
        "recordings": {20: "primary"},
    },
}


def build_block_paths(d, pos_label, rec_num):
    """Build LDV, MIC-LEFT, MIC-RIGHT paths for block recordings."""
    directory = d["dir"]
    ldv = f"{directory}/0223-LDV-40-boy({pos_label})-{rec_num}-block.wav"
    mic_l = f"{directory}/0223-MIC-LEFT-40-boy({pos_label})-{rec_num}-block.wav"
    mic_r = f"{directory}/0223-MIC-RIGHT-40-boy({pos_label})-{rec_num}-block.wav"
    return ldv, mic_l, mic_r


def build_unblock_paths(d, pos_label, rec_num):
    """Build MIC-LEFT, MIC-RIGHT paths for unblock recordings (no LDV)."""
    directory = d["dir"]
    mic_l = f"{directory}/0223-MIC-LEFT-40-boy({pos_label})-{rec_num}-unblock.wav"
    mic_r = f"{directory}/0223-MIC-RIGHT-40-boy({pos_label})-{rec_num}-unblock.wav"
    return mic_l, mic_r


# ============================================================
# Compute true angles
# ============================================================
positions = ["+0.0m", "+0.4m", "+0.8m", "-0.4m", "-0.8m"]
spk_x_map = {"+0.0m": 0.0, "+0.4m": 0.4, "+0.8m": 0.8, "-0.4m": -0.4, "-0.8m": -0.8}

print("=" * 70)
print("TRUE ANGLES (from geometry)")
print("=" * 70)
true_angles = {}
true_taus = {}
for pos in positions:
    spk_x = spk_x_map[pos]
    angle, tau = compute_true_angle(spk_x)
    true_angles[pos] = angle
    true_taus[pos] = tau
    print(f"  {pos:>6s}: spk_x={spk_x:+.1f}  theta_true = {angle:+8.3f} deg  tau_true = {tau:+8.4f} ms")
print()

# ============================================================
# Compute all results
# ============================================================
search_windows = [2.5, 4.08]

# Storage: results[condition][window][position] = (doa, tau_ms, err, rec_num)
# For S3-joint: results_s3[position] = (doa, delta_tau_ms, err, rec_num)
results_unblock = {w: {} for w in search_windows}
results_block = {w: {} for w in search_windows}
results_s3 = {}

# Also store alternate recordings for +-0.8m
results_block_alt = {w: {} for w in search_windows}
results_s3_alt = {}

# Additional block +0.0m #22
results_block_alt_00 = {w: {} for w in search_windows}
results_s3_alt_00 = {}

print("=" * 70)
print("PROCESSING UNBLOCK MIC-MIC")
print("=" * 70)
for pos in positions:
    d = unblock_files[pos]
    spk_x = d["spk_x"]
    for rec_num in d["recordings"]:
        mic_l_path, mic_r_path = build_unblock_paths(d, pos, rec_num)
        print(f"\n  {pos} rec#{rec_num}:")
        print(f"    L: {os.path.basename(mic_l_path)}")
        print(f"    R: {os.path.basename(mic_r_path)}")

        # Check files exist
        for p in [mic_l_path, mic_r_path]:
            if not os.path.exists(p):
                print(f"    *** MISSING: {p}")

        sr_l, sig_l = load_wav(mic_l_path)
        sr_r, sig_r = load_wav(mic_r_path)
        assert sr_l == sr_r == fs_expected, f"Unexpected sample rate: {sr_l}, {sr_r}"

        # Trim to same length
        min_len = min(len(sig_l), len(sig_r))
        sig_l = sig_l[:min_len]
        sig_r = sig_r[:min_len]

        for w in search_windows:
            tau_ms, peak_val, _, _, top5 = gcc_phat(sig_l, sig_r, fs_expected, w)
            doa = tau_to_doa(tau_ms)
            err = abs(doa - true_angles[pos])
            results_unblock[w][pos] = (doa, tau_ms, err, rec_num)
            print(f"    Window +-{w}ms: tau={tau_ms:+.4f}ms  DoA={doa:+.3f}deg  |err|={err:.3f}deg")


print("\n" + "=" * 70)
print("PROCESSING BLOCK MIC-MIC & S3-JOINT")
print("=" * 70)
for pos in positions:
    d = block_files[pos]
    spk_x = d["spk_x"]
    for rec_num, rec_type in d["recordings"].items():
        ldv_path, mic_l_path, mic_r_path = build_block_paths(d, pos, rec_num)
        print(f"\n  {pos} rec#{rec_num} ({rec_type}):")
        print(f"    LDV: {os.path.basename(ldv_path)}")
        print(f"    L:   {os.path.basename(mic_l_path)}")
        print(f"    R:   {os.path.basename(mic_r_path)}")

        for p in [ldv_path, mic_l_path, mic_r_path]:
            if not os.path.exists(p):
                print(f"    *** MISSING: {p}")

        sr_ldv, sig_ldv = load_wav(ldv_path)
        sr_l, sig_l = load_wav(mic_l_path)
        sr_r, sig_r = load_wav(mic_r_path)
        assert sr_ldv == sr_l == sr_r == fs_expected

        min_len = min(len(sig_ldv), len(sig_l), len(sig_r))
        sig_ldv = sig_ldv[:min_len]
        sig_l = sig_l[:min_len]
        sig_r = sig_r[:min_len]

        # MIC-MIC GCC-PHAT
        for w in search_windows:
            tau_ms, peak_val, _, _, top5 = gcc_phat(sig_l, sig_r, fs_expected, w)
            doa = tau_to_doa(tau_ms)
            err = abs(doa - true_angles[pos])

            if rec_type == "primary":
                results_block[w][pos] = (doa, tau_ms, err, rec_num)
            else:
                if pos == "+0.0m":
                    results_block_alt_00[w][pos] = (doa, tau_ms, err, rec_num)
                else:
                    results_block_alt[w][pos] = (doa, tau_ms, err, rec_num)

            print(f"    MIC-MIC +-{w}ms: tau={tau_ms:+.4f}ms  DoA={doa:+.3f}deg  |err|={err:.3f}deg")

        # S3-joint
        doa_s3, dtau_s3, tau_vl_th, tau_vr_th = s3_joint(sig_ldv, sig_l, sig_r, fs_expected, spk_x)
        err_s3 = abs(doa_s3 - true_angles[pos])

        if rec_type == "primary":
            results_s3[pos] = (doa_s3, dtau_s3, err_s3, rec_num, tau_vl_th, tau_vr_th)
        else:
            if pos == "+0.0m":
                results_s3_alt_00[pos] = (doa_s3, dtau_s3, err_s3, rec_num, tau_vl_th, tau_vr_th)
            else:
                results_s3_alt[pos] = (doa_s3, dtau_s3, err_s3, rec_num, tau_vl_th, tau_vr_th)

        print(f"    S3-joint: DoA={doa_s3:+.3f}deg  |err|={err_s3:.3f}deg  delta_tau={dtau_s3:+.4f}ms")
        print(f"             tau_VL_theory={tau_vl_th:.4f}ms  tau_VR_theory={tau_vr_th:.4f}ms")


# ============================================================
# Print formatted tables
# ============================================================

def print_table_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# --- Table A: +-2.5ms ---
print_table_header("TABLE A: Current Paper Table 1 (+-2.5ms search window)")
w = 2.5
hdr = f"{'Pos':>6s} | {'True':>8s} | {'Unblk DoA':>10s} {'|err|':>7s} | {'Block DoA':>10s} {'|err|':>7s} | {'S3 DoA':>10s} {'|err|':>7s}"
print(hdr)
print("-" * len(hdr))
mae_unblk = []
mae_blk = []
mae_s3 = []
for pos in positions:
    theta_true = true_angles[pos]

    ub = results_unblock[w].get(pos)
    bl = results_block[w].get(pos)
    s3 = results_s3.get(pos)

    ub_str = f"{ub[0]:+8.3f}  {ub[2]:6.3f}" if ub else "   N/A      N/A"
    bl_str = f"{bl[0]:+8.3f}  {bl[2]:6.3f}" if bl else "   N/A      N/A"
    s3_str = f"{s3[0]:+8.3f}  {s3[2]:6.3f}" if s3 else "   N/A      N/A"

    if ub: mae_unblk.append(ub[2])
    if bl: mae_blk.append(bl[2])
    if s3: mae_s3.append(s3[2])

    print(f"{pos:>6s} | {theta_true:+8.3f} | {ub_str} | {bl_str} | {s3_str}")

print("-" * len(hdr))
print(f"{'MAE':>6s} | {'':>8s} | {'':>10s} {np.mean(mae_unblk):6.3f}  | {'':>10s} {np.mean(mae_blk):6.3f}  | {'':>10s} {np.mean(mae_s3):6.3f}")


# --- Table B: +-4.08ms ---
print_table_header("TABLE B: Proposed Update (+-4.08ms search window)")
w = 4.08
hdr = f"{'Pos':>6s} | {'True':>8s} | {'Unblk DoA':>10s} {'|err|':>7s} | {'Block DoA':>10s} {'|err|':>7s} | {'S3 DoA':>10s} {'|err|':>7s}"
print(hdr)
print("-" * len(hdr))
mae_unblk = []
mae_blk = []
mae_s3 = []
for pos in positions:
    theta_true = true_angles[pos]

    ub = results_unblock[w].get(pos)
    bl = results_block[w].get(pos)
    s3 = results_s3.get(pos)

    ub_str = f"{ub[0]:+8.3f}  {ub[2]:6.3f}" if ub else "   N/A      N/A"
    bl_str = f"{bl[0]:+8.3f}  {bl[2]:6.3f}" if bl else "   N/A      N/A"
    s3_str = f"{s3[0]:+8.3f}  {s3[2]:6.3f}" if s3 else "   N/A      N/A"

    if ub: mae_unblk.append(ub[2])
    if bl: mae_blk.append(bl[2])
    if s3: mae_s3.append(s3[2])

    print(f"{pos:>6s} | {theta_true:+8.3f} | {ub_str} | {bl_str} | {s3_str}")

print("-" * len(hdr))
print(f"{'MAE':>6s} | {'':>8s} | {'':>10s} {np.mean(mae_unblk):6.3f}  | {'':>10s} {np.mean(mae_blk):6.3f}  | {'':>10s} {np.mean(mae_s3):6.3f}")


# --- Table C: Non-determinism for +-0.8m (and +0.0m #22) ---
print_table_header("TABLE C: Non-determinism — Alternate Recordings")
print(f"\n  Positions with multiple recordings: +0.0m (#18 vs #22), +0.8m (#17 vs #21), -0.8m (#20 vs #21)")
print()

for w in search_windows:
    print(f"  --- Window +-{w}ms ---")
    hdr2 = f"  {'Pos':>6s} | {'True':>8s} | {'Rec#':>4s} {'DoA':>10s} {'|err|':>7s} | {'Rec#':>4s} {'DoA':>10s} {'|err|':>7s} | {'Delta DoA':>10s}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    # +0.0m
    pos = "+0.0m"
    theta_true = true_angles[pos]
    bl_prim = results_block[w].get(pos)
    bl_alt = results_block_alt_00[w].get(pos)
    if bl_prim and bl_alt:
        delta = bl_alt[0] - bl_prim[0]
        print(f"  {pos:>6s} | {theta_true:+8.3f} | #{bl_prim[3]:>2d} {bl_prim[0]:+8.3f}  {bl_prim[2]:6.3f} | #{bl_alt[3]:>2d} {bl_alt[0]:+8.3f}  {bl_alt[2]:6.3f} | {delta:+8.3f}")

    # +0.8m
    pos = "+0.8m"
    theta_true = true_angles[pos]
    bl_prim = results_block[w].get(pos)
    bl_alt = results_block_alt[w].get(pos)
    if bl_prim and bl_alt:
        delta = bl_alt[0] - bl_prim[0]
        print(f"  {pos:>6s} | {theta_true:+8.3f} | #{bl_prim[3]:>2d} {bl_prim[0]:+8.3f}  {bl_prim[2]:6.3f} | #{bl_alt[3]:>2d} {bl_alt[0]:+8.3f}  {bl_alt[2]:6.3f} | {delta:+8.3f}")

    # -0.8m
    pos = "-0.8m"
    theta_true = true_angles[pos]
    bl_prim = results_block[w].get(pos)
    bl_alt = results_block_alt[w].get(pos)
    if bl_prim and bl_alt:
        delta = bl_alt[0] - bl_prim[0]
        print(f"  {pos:>6s} | {theta_true:+8.3f} | #{bl_prim[3]:>2d} {bl_prim[0]:+8.3f}  {bl_prim[2]:6.3f} | #{bl_alt[3]:>2d} {bl_alt[0]:+8.3f}  {bl_alt[2]:6.3f} | {delta:+8.3f}")

    print()

# S3-joint non-determinism
print(f"  --- S3-Joint (window-independent) ---")
hdr2 = f"  {'Pos':>6s} | {'True':>8s} | {'Rec#':>4s} {'DoA':>10s} {'|err|':>7s} | {'Rec#':>4s} {'DoA':>10s} {'|err|':>7s} | {'Delta DoA':>10s}"
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))

pos = "+0.0m"
theta_true = true_angles[pos]
s3_prim = results_s3.get(pos)
s3_alt = results_s3_alt_00.get(pos)
if s3_prim and s3_alt:
    delta = s3_alt[0] - s3_prim[0]
    print(f"  {pos:>6s} | {theta_true:+8.3f} | #{s3_prim[3]:>2d} {s3_prim[0]:+8.3f}  {s3_prim[2]:6.3f} | #{s3_alt[3]:>2d} {s3_alt[0]:+8.3f}  {s3_alt[2]:6.3f} | {delta:+8.3f}")

pos = "+0.8m"
theta_true = true_angles[pos]
s3_prim = results_s3.get(pos)
s3_alt = results_s3_alt.get(pos)
if s3_prim and s3_alt:
    delta = s3_alt[0] - s3_prim[0]
    print(f"  {pos:>6s} | {theta_true:+8.3f} | #{s3_prim[3]:>2d} {s3_prim[0]:+8.3f}  {s3_prim[2]:6.3f} | #{s3_alt[3]:>2d} {s3_alt[0]:+8.3f}  {s3_alt[2]:6.3f} | {delta:+8.3f}")

pos = "-0.8m"
theta_true = true_angles[pos]
s3_prim = results_s3.get(pos)
s3_alt = results_s3_alt.get(pos)
if s3_prim and s3_alt:
    delta = s3_alt[0] - s3_prim[0]
    print(f"  {pos:>6s} | {theta_true:+8.3f} | #{s3_prim[3]:>2d} {s3_prim[0]:+8.3f}  {s3_prim[2]:6.3f} | #{s3_alt[3]:>2d} {s3_alt[0]:+8.3f}  {s3_alt[2]:6.3f} | {delta:+8.3f}")


# --- Table D: S3-joint detail ---
print_table_header("TABLE D: S3-Joint Detailed Results (all primary recordings)")
hdr3 = f"{'Pos':>6s} | {'spk_x':>6s} | {'True':>8s} | {'S3 DoA':>10s} | {'|err|':>7s} | {'dtau':>8s} | {'tau_VL_th':>10s} | {'tau_VR_th':>10s}"
print(hdr3)
print("-" * len(hdr3))
for pos in positions:
    theta_true = true_angles[pos]
    spk_x = spk_x_map[pos]
    s3 = results_s3.get(pos)
    if s3:
        print(f"{pos:>6s} | {spk_x:+5.1f}  | {theta_true:+8.3f} | {s3[0]:+8.3f}   | {s3[2]:6.3f} | {s3[1]:+7.4f} | {s3[4]:+9.4f}  | {s3[5]:+9.4f}")


# ============================================================
# Summary comparison: 2.5ms vs 4.08ms
# ============================================================
print_table_header("SUMMARY: Window Comparison (+-2.5ms vs +-4.08ms)")
print(f"\n  {'':>6s} | {'--- +-2.5ms ---':^35s} | {'--- +-4.08ms ---':^35s}")
hdr4 = f"  {'Pos':>6s} | {'Unblk':>8s} {'Blk':>8s} {'S3':>8s} | {'Unblk':>8s} {'Blk':>8s} {'S3':>8s} | {'Notes'}"
print(hdr4)
print("  " + "-" * (len(hdr4) - 2))

for pos in positions:
    ub25 = results_unblock[2.5].get(pos)
    bl25 = results_block[2.5].get(pos)
    s3_r = results_s3.get(pos)
    ub408 = results_unblock[4.08].get(pos)
    bl408 = results_block[4.08].get(pos)

    ub25_doa = f"{ub25[0]:+7.2f}" if ub25 else "   N/A"
    bl25_doa = f"{bl25[0]:+7.2f}" if bl25 else "   N/A"
    s3_doa = f"{s3_r[0]:+7.2f}" if s3_r else "   N/A"
    ub408_doa = f"{ub408[0]:+7.2f}" if ub408 else "   N/A"
    bl408_doa = f"{bl408[0]:+7.2f}" if bl408 else "   N/A"

    # Note if result changed between windows
    changed = ""
    if ub25 and ub408 and abs(ub25[0] - ub408[0]) > 0.01:
        changed += f"Unblk changed! "
    if bl25 and bl408 and abs(bl25[0] - bl408[0]) > 0.01:
        changed += f"Blk changed! "

    print(f"  {pos:>6s} | {ub25_doa:>8s} {bl25_doa:>8s} {s3_doa:>8s} | {ub408_doa:>8s} {bl408_doa:>8s} {s3_doa:>8s} | {changed}")

# MAE comparison
print()
for w in search_windows:
    errs_ub = [results_unblock[w][pos][2] for pos in positions if pos in results_unblock[w]]
    errs_bl = [results_block[w][pos][2] for pos in positions if pos in results_block[w]]
    errs_s3 = [results_s3[pos][2] for pos in positions if pos in results_s3]
    print(f"  MAE +-{w}ms:  Unblock={np.mean(errs_ub):.3f}  Block={np.mean(errs_bl):.3f}  S3-joint={np.mean(errs_s3):.3f}")


# ============================================================
# Raw tau values for debugging
# ============================================================
print_table_header("RAW TAU VALUES (for debugging)")
for w in search_windows:
    print(f"\n  --- Window +-{w}ms ---")
    print(f"  {'Pos':>6s} | {'Cond':>8s} | {'tau(ms)':>10s} | {'DoA(deg)':>10s} | {'True(deg)':>10s} | {'|err|':>7s} | {'Rec#':>4s}")
    print("  " + "-" * 80)
    for pos in positions:
        theta_true = true_angles[pos]
        ub = results_unblock[w].get(pos)
        bl = results_block[w].get(pos)
        if ub:
            print(f"  {pos:>6s} | {'Unblock':>8s} | {ub[1]:+9.4f}  | {ub[0]:+9.3f}  | {theta_true:+9.3f}  | {ub[2]:6.3f} | #{ub[3]}")
        if bl:
            print(f"  {pos:>6s} | {'Block':>8s} | {bl[1]:+9.4f}  | {bl[0]:+9.3f}  | {theta_true:+9.3f}  | {bl[2]:6.3f} | #{bl[3]}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

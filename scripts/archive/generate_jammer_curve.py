import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.fft as sfft

def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return sr, data

def gcc_phat(sig1, sig2, fs, max_tau=0.010):
    n = len(sig1) + len(sig2)
    fast_n = sfft.next_fast_len(n)
    SIG1 = sfft.fft(sig1, fast_n)
    SIG2 = sfft.fft(sig2, fast_n)
    R = SIG1 * np.conj(SIG2)
    R = R / (np.abs(R) + 1e-10)
    cc = np.real(sfft.ifft(R))

    max_shift = int(max_tau * fs)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    abs_cc = np.abs(cc)
    
    peak_idx = int(np.argmax(abs_cc))
    
    if 0 < peak_idx < len(abs_cc) - 1:
        y0 = abs_cc[peak_idx - 1]
        y1 = abs_cc[peak_idx]
        y2 = abs_cc[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            shift = 0.5 * (y0 - y2) / denom
        else:
            shift = 0.0
    else:
        shift = 0.0

    tau = ((peak_idx - max_shift) + shift) / fs
    
    # PSR
    mask = np.ones_like(abs_cc, dtype=bool)
    exclude = 50
    lo = max(0, peak_idx - exclude)
    hi = min(len(abs_cc), peak_idx + exclude + 1)
    mask[lo:hi] = False
    sidelobe_max = abs_cc[mask].max() if np.any(mask) else 0.0
    psr = 20 * np.log10(abs_cc[peak_idx] / (sidelobe_max + 1e-10))
    
    return float(tau), float(psr)

def main():
    root = "/home/sbplab/jiawei/0222-block/0223-block/"
    
    f_ldv = os.path.join(root, "0223-LDV-40-boy(+0.4m)-13-block.wav")
    f_micL = os.path.join(root, "0223-MIC-LEFT-40-boy(+0.4m)-13-block.wav")
    f_micR = os.path.join(root, "0223-MIC-RIGHT-40-boy(+0.4m)-13-block.wav")
    
    f_jamL = os.path.join(root, "0223-unblock-7(high)/0223-MIC-LEFT-40-boy(-0.8m)-20-unblock.wav")
    f_jamR = os.path.join(root, "0223-unblock-7(high)/0223-MIC-RIGHT-40-boy(-0.8m)-20-unblock.wav")

    fs, tgt_ldv = load_wav(f_ldv)
    _, tgt_micL = load_wav(f_micL)
    _, tgt_micR = load_wav(f_micR)
    
    _, jam_micL = load_wav(f_jamL)
    _, jam_micR = load_wav(f_jamR)
    
    n_samples = int(5.0 * fs)
    tgt_ldv = tgt_ldv[:n_samples]
    tgt_micL = tgt_micL[:n_samples]
    tgt_micR = tgt_micR[:n_samples]
    jam_micL = jam_micL[:n_samples]
    jam_micR = jam_micR[:n_samples]
    
    tgt_rms = np.sqrt(np.mean((tgt_micL)**2 + (tgt_micR)**2))
    jam_rms = np.sqrt(np.mean((jam_micL)**2 + (jam_micR)**2))
    
    c = 343.0
    micL_pos = np.array([-0.7, 2.0])
    micR_pos = np.array([0.7, 2.0])
    ldv_pos = np.array([0.0, 0.5])
    tgt_pos = np.array([0.4, 0.0])
    
    tau_true = (np.linalg.norm(tgt_pos - micL_pos) - np.linalg.norm(tgt_pos - micR_pos)) / c
    
    sjr_list = np.linspace(-40, 20, 31)
    
    mae_mic_only = []
    mae_pi_gs = []
    
    for k, sjr in enumerate(sjr_list):
        scale = (tgt_rms / jam_rms) * (10.0 ** (-sjr / 20.0))
        
        mix_micL = tgt_micL + scale * jam_micL
        mix_micR = tgt_micR + scale * jam_micR
        
        # 1. Mic-Mic GCC-PHAT
        tau_mic, _ = gcc_phat(mix_micL, mix_micR, fs)
        mae_mic_only.append(abs(tau_mic - tau_true) * c / 1.4 * 180 / np.pi) 
        
        # 2. PI-GS
        X_range = np.linspace(-1.0, 1.0, 201)
        best_X = 0
        best_score = -np.inf
        
        n = len(tgt_ldv) + len(mix_micL)
        fast_n = sfft.next_fast_len(n)
        
        SIG_V = sfft.fft(tgt_ldv, fast_n)
        SIG_L = sfft.fft(mix_micL, fast_n)
        SIG_R = sfft.fft(mix_micR, fast_n)
        
        R_VL = SIG_V * np.conj(SIG_L)
        R_VL = np.real(sfft.ifft(R_VL / (np.abs(R_VL) + 1e-10)))
        
        R_VR = SIG_V * np.conj(SIG_R)
        R_VR = np.real(sfft.ifft(R_VR / (np.abs(R_VR) + 1e-10)))
        
        def get_val(R, tau, fs, fast_n):
            shift = int(np.round(tau * fs))
            return R[shift] if shift >= 0 else R[fast_n + shift]
            
        for X in X_range:
            p = np.array([X, 0.0])
            t_V = np.linalg.norm(p - ldv_pos) / c
            t_L = np.linalg.norm(p - micL_pos) / c
            t_R = np.linalg.norm(p - micR_pos) / c
            
            tau_VL = t_L - t_V
            tau_VR = t_R - t_V
            
            score = get_val(R_VL, tau_VL, fs, fast_n) + get_val(R_VR, tau_VR, fs, fast_n)
            if score > best_score:
                best_score = score
                best_X = X
                
        tau_pi = (np.linalg.norm(np.array([best_X, 0.0]) - micL_pos) - np.linalg.norm(np.array([best_X, 0.0]) - micR_pos)) / c
        mae_pi_gs.append(abs(tau_pi - tau_true) * c / 1.4 * 180 / np.pi)

    # Print data for PGFPlots
    print("--- PGFPlots Data: Jammer Resilience Curve ---")
    print("SJR_dB\tMic-Mic_MAE\tPI-GS_MAE")
    for s, m, p in zip(sjr_list, mae_mic_only, mae_pi_gs):
        print(f"{s:.1f}\t{m:.3f}\t{p:.3f}")
    
    # Save to a generic dat file for pgfplots
    out_dat = '/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/jammer_resilience_curve.dat'
    with open(out_dat, 'w') as f:
        f.write("SJR_dB\tMic-Mic_MAE\tPI-GS_MAE\n")
        for s, m, p in zip(sjr_list, mae_mic_only, mae_pi_gs):
            f.write(f"{s:.1f}\t{m:.3f}\t{p:.3f}\n")
    print(f"Data saved to {out_dat}")

if __name__ == '__main__':
    main()

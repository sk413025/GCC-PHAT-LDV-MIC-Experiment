import os
import sys
import numpy as np
import json
sys.path.append('/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/scripts')
import stage4_doa_ldv_vs_mic_comparison as stage4
from scipy.signal import stft, istft

def main():
    root = "/home/sbplab/jiawei/0222-block/0223-block/"
    
    f_ldv = os.path.join(root, "0223-LDV-40-boy(+0.4m)-13-block.wav")
    f_micL = os.path.join(root, "0223-MIC-LEFT-40-boy(+0.4m)-13-block.wav")
    f_micR = os.path.join(root, "0223-MIC-RIGHT-40-boy(+0.4m)-13-block.wav")
    
    f_jamL = os.path.join(root, "0223-unblock-7(high)/0223-MIC-LEFT-40-boy(-0.8m)-20-unblock.wav")
    f_jamR = os.path.join(root, "0223-unblock-7(high)/0223-MIC-RIGHT-40-boy(-0.8m)-20-unblock.wav")

    fs, tgt_ldv = stage4.load_wav(f_ldv)
    _, tgt_micL = stage4.load_wav(f_micL)
    _, tgt_micR = stage4.load_wav(f_micR)
    
    _, jam_micL = stage4.load_wav(f_jamL)
    _, jam_micR = stage4.load_wav(f_jamR)
    
    # 5 seconds
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
    
    config = stage4.DEFAULT_CONFIG.copy()
    config["ldv_prealign"] = "omp"
    config["alignment_mode"] = "omp"
    
    # Align LDV to micL using OMP (this is what stage4 does for ldv_micl)
    print("Running OMP alignment on full 5s window...")
    f, t, Zxx_ldv = stft(tgt_ldv, fs, nperseg=config["n_fft"], noverlap=config["n_fft"] - config["hop_length"])
    _, _, Zxx_micL = stft(tgt_micL, fs, nperseg=config["n_fft"], noverlap=config["n_fft"] - config["hop_length"])
    
    # Process OMP in chunks if needed, but since it's 5s we can do it all at once if start_t=0 and tw is full length
    # Wait, apply_omp_alignment expects a start_t and tw!
    # Let's just use the STFT dimensions
    config["tw"] = Zxx_ldv.shape[1]
    Zxx_omp = stage4.apply_omp_alignment(Zxx_ldv, Zxx_micL, config, 0)
    _, ldv_aligned = istft(Zxx_omp, fs, nperseg=config["n_fft"], noverlap=config["n_fft"] - config["hop_length"])
    
    # Ensure length matches
    ldv_aligned = ldv_aligned[:n_samples]
    
    sjr_list = np.linspace(-40, 20, 13) # reducing resolution to speed up
    
    mae_mic_only = []
    mae_pi_gs = []
    
    print("SJR_dB\tMic-Mic_MAE\tPI-GS_MAE")
    for sjr in sjr_list:
        scale = (tgt_rms / jam_rms) * (10.0 ** (-sjr / 20.0))
        
        mix_micL = tgt_micL + scale * jam_micL
        mix_micR = tgt_micR + scale * jam_micR
        
        # Mic-Mic Baseline
        res_mic = stage4.estimate_tdoa_gcc_phat(
            mix_micL, mix_micR, fs,
            max_lag_samples=int(config["gcc_max_lag_ms"] * fs / 1000.0),
            bandpass=(500.0, 2000.0),
            psr_exclude_samples=config["psr_exclude_samples"],
            guided_tau_ms=None,
            guided_radius_ms=None
        )
        tau_mic = res_mic["tau_ms"] / 1000.0
        err_mic = abs(tau_mic - tau_true) * c / 1.4 * 180 / np.pi
        mae_mic_only.append(err_mic)
        
        # PI-GS (S3-Joint) with aligned LDV
        # stage4 doesn't have an explicit S3-Joint 2D scan function, we write it here
        # but using bandpass!
        ldv_bp = stage4.bandpass_filter(ldv_aligned, 500.0, 2000.0, fs)
        micL_bp = stage4.bandpass_filter(mix_micL, 500.0, 2000.0, fs)
        micR_bp = stage4.bandpass_filter(mix_micR, 500.0, 2000.0, fs)
        
        n = len(ldv_bp) + len(micL_bp)
        import scipy.fft as sfft
        fast_n = sfft.next_fast_len(n)
        
        SIG_V = sfft.fft(ldv_bp, fast_n)
        SIG_L = sfft.fft(micL_bp, fast_n)
        SIG_R = sfft.fft(micR_bp, fast_n)
        
        R_VL = SIG_V * np.conj(SIG_L)
        R_VL = np.real(sfft.ifft(R_VL / (np.abs(R_VL) + 1e-10)))
        
        R_VR = SIG_V * np.conj(SIG_R)
        R_VR = np.real(sfft.ifft(R_VR / (np.abs(R_VR) + 1e-10)))
        
        def get_val(R, tau, fs, fast_n):
            shift = int(np.round(tau * fs))
            return R[shift] if shift >= 0 else R[fast_n + shift]
            
        X_range = np.linspace(-1.0, 1.0, 201)
        best_X = 0
        best_score = -np.inf
        
        for X in X_range:
            p = np.array([X, 0.0])
            t_V = np.linalg.norm(p - ldv_pos) / c
            t_L = np.linalg.norm(p - micL_pos) / c
            t_R = np.linalg.norm(p - micR_pos) / c
            
            # Since LDV is already aligned to MicL, tau_V is effectively tau_L
            # So the empirical delay from ldv_aligned to MicL is 0.
            # R_VL peak should be at 0.
            # Theoretical delay difference from Source to (MicL vs aligned_LDV):
            # The aligned LDV acts as a virtual sensor AT MicL's position in terms of time!
            # Wait, OMP alignment shifts LDV to match MicL's phase.
            # So ldv_aligned has the exact same propagation delay as MicL.
            # Thus, tau_VL_theory = 0 always!
            # And tau_VR_theory = tau_R - tau_L.
            tau_VL_theory = 0.0
            tau_VR_theory = t_R - t_L
            
            score = get_val(R_VL, tau_VL_theory, fs, fast_n) + get_val(R_VR, tau_VR_theory, fs, fast_n)
            if score > best_score:
                best_score = score
                best_X = X
                
        tau_pi = (np.linalg.norm(np.array([best_X, 0.0]) - micL_pos) - np.linalg.norm(np.array([best_X, 0.0]) - micR_pos)) / c
        err_pi = abs(tau_pi - tau_true) * c / 1.4 * 180 / np.pi
        mae_pi_gs.append(err_pi)
        
        print(f"{sjr:.1f}\t{err_mic:.3f}\t{err_pi:.3f}")

if __name__ == '__main__':
    main()

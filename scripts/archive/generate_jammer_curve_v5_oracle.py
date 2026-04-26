import os
import sys
import numpy as np
import json

sys.path.append('/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/scripts')
import train_pi_dnn_s3joint_comparison as s3joint
from scipy.io import wavfile

c = 343.0
d_mic = 1.4

def get_true_theta(spk_x):
    d_SL = np.sqrt((spk_x + 0.7)**2 + 2.0**2)
    d_SR = np.sqrt((spk_x - 0.7)**2 + 2.0**2)
    tau_mic_ms = (d_SL - d_SR) / c * 1000
    theta_true = np.degrees(np.arcsin(np.clip(tau_mic_ms / 1000 * c / d_mic, -1, 1)))
    return theta_true

def compute_frame_wise_angles_oracle(ldv, micL, micR, spk_x, fs, win_sec=0.5, stride_sec=0.25):
    """
    Computes frame-wise DoA angles.
    Mic-Mic Baseline: Unconstrained 1D search
    PI-GS (Oracle): 1D search tightly constrained around the theoretical tau for spk_x
    """
    win_samples = int(win_sec * fs)
    stride_samples = int(stride_sec * fs)
    n_windows = (len(ldv) - win_samples) // stride_samples + 1
    
    thetas_mic = []
    thetas_pi = []
    
    max_abs_ml = np.max(np.abs(micL))
    
    for w in range(n_windows):
        start = w * stride_samples
        end = start + win_samples
        
        s_ldv = ldv[start:end]
        s_mL = micL[start:end]
        s_mR = micR[start:end]
        
        if np.max(np.abs(s_mL)) < max_abs_ml * 0.1:
            continue
            
        # Mic-Mic Baseline (Unconstrained within geometric bounds)
        g_lr, lags_lr = s3joint.gcc_phat(s_mL, s_mR, fs)
        mask_lr = (lags_lr * 1000 >= -4.1) & (lags_lr * 1000 <= 4.1)
        valid_lags = lags_lr[mask_lr]
        valid_g = g_lr[mask_lr]
        if len(valid_lags) > 0:
            dt_lr = valid_lags[np.argmax(valid_g)] * 1000
            theta_mic = np.degrees(np.arcsin(np.clip(dt_lr / 1000 * c / d_mic, -1, 1)))
            thetas_mic.append(theta_mic)
        
        # PI-GS Oracle (S3-joint 1D with explicitly provided spk_x)
        g_vl, lags_vl = s3joint.gcc_phat(s_ldv, s_mL, fs)
        g_vr, lags_vr = s3joint.gcc_phat(s_ldv, s_mR, fs)
        
        s3_res = s3joint.s3_joint(g_vl, lags_vl, g_vr, lags_vr, spk_x, hw=0.5)
        if s3_res is not None:
            _, _, dt_pi, _ = s3_res
            theta_pi = np.degrees(np.arcsin(np.clip(dt_pi / 1000 * c / d_mic, -1, 1)))
            thetas_pi.append(theta_pi)
            
    return thetas_mic, thetas_pi

def main():
    root = "/home/sbplab/jiawei/0222-block/0223-block/"
    
    f_ldv = os.path.join(root, "0223-block-3(high)", "0223-LDV-40-boy(+0.4m)-16-block.wav")
    f_micL = os.path.join(root, "0223-block-3(high)", "0223-MIC-LEFT-40-boy(+0.4m)-16-block.wav")
    f_micR = os.path.join(root, "0223-block-3(high)", "0223-MIC-RIGHT-40-boy(+0.4m)-16-block.wav")
    
    f_jamL = os.path.join(root, "0223-unblock-7(high)", "0223-MIC-LEFT-40-boy(-0.8m)-20-unblock.wav")
    f_jamR = os.path.join(root, "0223-unblock-7(high)", "0223-MIC-RIGHT-40-boy(-0.8m)-20-unblock.wav")

    fs, tgt_ldv = wavfile.read(f_ldv)
    _, tgt_micL = wavfile.read(f_micL)
    _, tgt_micR = wavfile.read(f_micR)
    
    _, jam_micL = wavfile.read(f_jamL)
    _, jam_micR = wavfile.read(f_jamR)
    
    # Extract the pure speech segment (3s to 8s) to avoid the Chirp segment which artificially helps Mic-Mic
    start_idx = int(3.0 * fs)
    end_idx = int(8.0 * fs)
    
    tgt_ldv = tgt_ldv[start_idx:end_idx].astype(np.float64)
    tgt_micL = tgt_micL[start_idx:end_idx].astype(np.float64)
    tgt_micR = tgt_micR[start_idx:end_idx].astype(np.float64)
    jam_micL = jam_micL[start_idx:end_idx].astype(np.float64)
    jam_micR = jam_micR[start_idx:end_idx].astype(np.float64)
    
    tgt_ldv -= np.mean(tgt_ldv)
    tgt_micL -= np.mean(tgt_micL)
    tgt_micR -= np.mean(tgt_micR)
    jam_micL -= np.mean(jam_micL)
    jam_micR -= np.mean(jam_micR)
    
    tgt_rms = np.sqrt(np.mean((tgt_micL)**2 + (tgt_micR)**2))
    # Emulate the receiving-room jammer power based on the unblocked jammer recording
    jam_rms = np.sqrt(np.mean((jam_micL)**2 + (jam_micR)**2))
    
    tgt_spk_x = 0.4
    theta_true = get_true_theta(tgt_spk_x)
    print(f"Target spk_x: {tgt_spk_x:+.1f}m -> theta: {theta_true:.2f} deg")
    
    sjr_list = np.linspace(-40, 20, 31)
    
    out_dat = '/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/jammer_resilience_curve_s3joint_oracle.dat'
    with open(out_dat, 'w') as f:
        f.write("SJR_dB\tMic-Mic_MAE\tPI-GS_MAE\n")
        
        for sjr in sjr_list:
            scale = (tgt_rms / jam_rms) * (10.0 ** (-sjr / 20.0))
            
            mix_micL = tgt_micL + scale * jam_micL
            mix_micR = tgt_micR + scale * jam_micR
            
            # Post-mix centering
            mix_micL -= np.mean(mix_micL)
            mix_micR -= np.mean(mix_micR)
            
            # Compute frame-wise angles with the ground truth spk_x passed to the oracle
            thetas_mic, thetas_pi = compute_frame_wise_angles_oracle(
                tgt_ldv, mix_micL, mix_micR, tgt_spk_x, fs
            )
            
            theta_mic_med = np.median(thetas_mic) if thetas_mic else float('nan')
            theta_pi_med = np.median(thetas_pi) if thetas_pi else float('nan')
                
            err_mic = abs(theta_mic_med - theta_true)
            err_pi = abs(theta_pi_med - theta_true)
            
            print(f"{sjr:5.1f}\tMic={theta_mic_med:6.2f}(E:{err_mic:5.2f})\tPI-GS={theta_pi_med:6.2f}(E:{err_pi:5.2f})")
            f.write(f"{sjr:.1f}\t{err_mic:.3f}\t{err_pi:.3f}\n")
            f.flush()

    print(f"Data saved to {out_dat}")

if __name__ == '__main__':
    main()

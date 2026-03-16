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
    my = 2.0 - 0.25
    d_SL = np.sqrt((spk_x + 0.7)**2 + 2.0**2)
    d_SR = np.sqrt((spk_x - 0.7)**2 + 2.0**2)
    tau_mic_ms = (d_SL - d_SR) / c * 1000
    theta_true = np.degrees(np.arcsin(np.clip(tau_mic_ms / 1000 * c / d_mic, -1, 1)))
    return theta_true

def block_process(sig1, sig2, fs, win_sec, stride_sec):
    win_samples = int(win_sec * fs)
    stride_samples = int(stride_sec * fs)
    n_windows = (len(sig1) - win_samples) // stride_samples + 1
    
    gvl_avg = None
    lags = None
    for w in range(n_windows):
        start = w * stride_samples
        end = start + win_samples
        s1 = sig1[start:end]
        s2 = sig2[start:end]
        g, l = s3joint.gcc_phat(s1, s2, fs)
        if gvl_avg is None:
            gvl_avg = np.zeros_like(g)
            lags = l
        gvl_avg += g
    if n_windows > 0:
        gvl_avg /= n_windows
    return gvl_avg, lags

def main():
    root = "/home/sbplab/jiawei/0222-block/0223-block/"
    
    f_ldv = os.path.join(root, "0223-block-3(high)", "0223-LDV-40-boy(+0.4m)-16-block.wav")
    f_micL = os.path.join(root, "0223-block-3(high)", "0223-MIC-LEFT-40-boy(+0.4m)-16-block.wav")
    f_micR = os.path.join(root, "0223-block-3(high)", "0223-MIC-RIGHT-40-boy(+0.4m)-16-block.wav")
    
    f_jamL = "/home/sbplab/jiawei/0222-block/0223-block-7(high)/0223-MIC-LEFT-40-boy(-0.8m)-20-block.wav"
    f_jamR = "/home/sbplab/jiawei/0222-block/0223-block-7(high)/0223-MIC-RIGHT-40-boy(-0.8m)-20-block.wav"

    # Use Unblocked jammer instead to mimic severe receiving-room interference? No, paper says "receiving room interference", meaning unblocked jammer.
    f_jamL = os.path.join(root, "0223-unblock-7(high)/0223-MIC-LEFT-40-boy(-0.8m)-20-unblock.wav")
    f_jamR = os.path.join(root, "0223-unblock-7(high)/0223-MIC-RIGHT-40-boy(-0.8m)-20-unblock.wav")

    fs, tgt_ldv = wavfile.read(f_ldv)
    _, tgt_micL = wavfile.read(f_micL)
    _, tgt_micR = wavfile.read(f_micR)
    
    _, jam_micL = wavfile.read(f_jamL)
    _, jam_micR = wavfile.read(f_jamR)
    
    # Just take 5 seconds to match the previous sweep
    n_samples = int(5.0 * fs)
    tgt_ldv = tgt_ldv[:n_samples].astype(np.float64)
    tgt_micL = tgt_micL[:n_samples].astype(np.float64)
    tgt_micR = tgt_micR[:n_samples].astype(np.float64)
    jam_micL = jam_micL[:n_samples].astype(np.float64)
    jam_micR = jam_micR[:n_samples].astype(np.float64)
    
    tgt_ldv -= np.mean(tgt_ldv)
    tgt_micL -= np.mean(tgt_micL)
    tgt_micR -= np.mean(tgt_micR)
    jam_micL -= np.mean(jam_micL)
    jam_micR -= np.mean(jam_micR)
    
    tgt_rms = np.sqrt(np.mean((tgt_micL)**2 + (tgt_micR)**2))
    jam_rms = np.sqrt(np.mean((jam_micL)**2 + (jam_micR)**2))
    
    theta_true = get_true_theta(0.4)
    print(f"Target theta: {theta_true:.2f} deg")
    
    sjr_list = np.linspace(-40, 20, 31)
    
    out_dat = '/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/jammer_resilience_curve_s3joint.dat'
    with open(out_dat, 'w') as f:
        f.write("SJR_dB\tMic-Mic_MAE\tPI-GS_MAE\n")
        
        for sjr in sjr_list:
            scale = (tgt_rms / jam_rms) * (10.0 ** (-sjr / 20.0))
            
            mix_micL = tgt_micL + scale * jam_micL
            mix_micR = tgt_micR + scale * jam_micR
            
            # Mic-Mic Baseline
            g_lr, lags_lr = block_process(mix_micL, mix_micR, fs, 0.5, 0.25)
            dt_lr = lags_lr[np.argmax(g_lr)] * 1000
            theta_mic = np.degrees(np.arcsin(np.clip(dt_lr / 1000 * c / d_mic, -1, 1)))
            err_mic = abs(theta_mic - theta_true)
            
            # PI-GS (S3-joint-2D)
            gcc_vl, lags_vl = block_process(tgt_ldv, mix_micL, fs, 0.5, 0.25)
            gcc_vr, lags_vr = block_process(tgt_ldv, mix_micR, fs, 0.5, 0.25)
            
            s3_2d = s3joint.s3_joint_2d(gcc_vl, lags_vl, gcc_vr, lags_vr, hw=0.5)
            if s3_2d:
                theta_pi = s3_2d['theta_source']
                err_pi = abs(theta_pi - theta_true)
            else:
                err_pi = float('nan')
            
            print(f"{sjr:.1f}\t{err_mic:.3f}\t{err_pi:.3f}")
            f.write(f"{sjr:.1f}\t{err_mic:.3f}\t{err_pi:.3f}\n")
            f.flush()

    print(f"Data saved to {out_dat}")

if __name__ == '__main__':
    main()

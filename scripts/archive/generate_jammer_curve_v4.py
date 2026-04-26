import os
import sys
import numpy as np
import json

sys.path.append('/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/scripts')
import train_pi_dnn_s3joint_comparison as s3joint
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

c = 343.0
d_mic = 1.4

def get_true_theta(spk_x):
    d_SL = np.sqrt((spk_x + 0.7)**2 + 2.0**2)
    d_SR = np.sqrt((spk_x - 0.7)**2 + 2.0**2)
    tau_mic_ms = (d_SL - d_SR) / c * 1000
    theta_true = np.degrees(np.arcsin(np.clip(tau_mic_ms / 1000 * c / d_mic, -1, 1)))
    return theta_true

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
    
    start_idx = int(3.0 * fs)
    end_idx = int(8.0 * fs)
    
    tgt_ldv = tgt_ldv[start_idx:end_idx].astype(np.float64)
    tgt_micL = tgt_micL[start_idx:end_idx].astype(np.float64)
    tgt_micR = tgt_micR[start_idx:end_idx].astype(np.float64)
    jam_micL = jam_micL[start_idx:end_idx].astype(np.float64)
    jam_micR = jam_micR[start_idx:end_idx].astype(np.float64)
    
    tgt_ldv -= np.mean(tgt_ldv)
    
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
            
            mix_micL -= np.mean(mix_micL)
            mix_micR -= np.mean(mix_micR)
            
            # Mic-Mic Baseline
            g_lr, lags_lr = s3joint.gcc_phat(mix_micL, mix_micR, fs)
            mask_lr = (lags_lr * 1000 >= -4.1) & (lags_lr * 1000 <= 4.1)
            valid_lags = lags_lr[mask_lr]
            valid_g = g_lr[mask_lr]
            dt_lr = valid_lags[np.argmax(valid_g)] * 1000
            theta_mic = np.degrees(np.arcsin(np.clip(dt_lr / 1000 * c / d_mic, -1, 1)))
            
            # PI-GS (S3-joint-2D)
            g_vl, lags_vl = s3joint.gcc_phat(tgt_ldv, mix_micL, fs)
            g_vr, lags_vr = s3joint.gcc_phat(tgt_ldv, mix_micR, fs)
            
            s3_2d = s3joint.s3_joint_2d(g_vl, lags_vl, g_vr, lags_vr, hw=0.5)
            if s3_2d:
                theta_pi = s3_2d['theta_source']
            else:
                theta_pi = float('nan')
                
            err_mic = abs(theta_mic - theta_true)
            err_pi = abs(theta_pi - theta_true)
            
            print(f"{sjr:5.1f}\tMic={theta_mic:6.2f}(E:{err_mic:5.2f})\tPI-GS={theta_pi:6.2f}(E:{err_pi:5.2f})")
            f.write(f"{sjr:.1f}\t{err_mic:.3f}\t{err_pi:.3f}\n")
            f.flush()

    print(f"Data saved to {out_dat}")

if __name__ == '__main__':
    main()

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

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def get_true_theta(spk_x):
    d_SL = np.sqrt((spk_x + 0.7)**2 + 2.0**2)
    d_SR = np.sqrt((spk_x - 0.7)**2 + 2.0**2)
    tau_mic_ms = (d_SL - d_SR) / c * 1000
    theta_true = np.degrees(np.arcsin(np.clip(tau_mic_ms / 1000 * c / d_mic, -1, 1)))
    return theta_true

def compute_frame_wise_angles(ldv, micL, micR, fs, win_sec=0.5, stride_sec=0.25):
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
        
        # Simple VAD: skip if mic signal is too quiet (less than 10% of peak)
        if np.max(np.abs(s_mL)) < max_abs_ml * 0.1:
            continue
            
        # Mic-Mic Baseline
        g_lr, lags_lr = s3joint.gcc_phat(s_mL, s_mR, fs)
        mask_lr = (lags_lr * 1000 >= -4.1) & (lags_lr * 1000 <= 4.1)
        valid_lags = lags_lr[mask_lr]
        valid_g = g_lr[mask_lr]
        dt_lr = valid_lags[np.argmax(valid_g)] * 1000
        theta_mic = np.degrees(np.arcsin(np.clip(dt_lr / 1000 * c / d_mic, -1, 1)))
        thetas_mic.append(theta_mic)
        
        # PI-GS (S3-joint-2D)
        g_vl, lags_vl = s3joint.gcc_phat(s_ldv, s_mL, fs)
        g_vr, lags_vr = s3joint.gcc_phat(s_ldv, s_mR, fs)
        
        s3_2d = s3joint.s3_joint_2d(g_vl, lags_vl, g_vr, lags_vr, hw=0.5)
        if s3_2d:
            thetas_pi.append(s3_2d['theta_source'])
            
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
    
    n_samples = int(5.0 * fs)
    tgt_ldv = tgt_ldv[:n_samples].astype(np.float64)
    tgt_micL = tgt_micL[:n_samples].astype(np.float64)
    tgt_micR = tgt_micR[:n_samples].astype(np.float64)
    jam_micL = jam_micL[:n_samples].astype(np.float64)
    jam_micR = jam_micR[:n_samples].astype(np.float64)
    
    # Remove DC offset to fix the lag 0.0 anomaly during silences
    tgt_ldv -= np.mean(tgt_ldv)
    tgt_micL -= np.mean(tgt_micL)
    tgt_micR -= np.mean(tgt_micR)
    jam_micL -= np.mean(jam_micL)
    jam_micR -= np.mean(jam_micR)
    
    tgt_ldv = butter_bandpass_filter(tgt_ldv, 500, 2000, fs)
    tgt_micL = butter_bandpass_filter(tgt_micL, 500, 2000, fs)
    tgt_micR = butter_bandpass_filter(tgt_micR, 500, 2000, fs)
    jam_micL = butter_bandpass_filter(jam_micL, 500, 2000, fs)
    jam_micR = butter_bandpass_filter(jam_micR, 500, 2000, fs)
    
    tgt_rms = np.sqrt(np.mean((tgt_micL)**2 + (tgt_micR)**2))
    # Evaluate jammer RMS only where it's actually active to prevent over-amplification
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
            
            thetas_mic, thetas_pi = compute_frame_wise_angles(tgt_ldv, mix_micL, mix_micR, fs)
            
            if not thetas_mic:
                theta_mic_med = float('nan')
            else:
                theta_mic_med = np.median(thetas_mic)
                
            if not thetas_pi:
                theta_pi_med = float('nan')
            else:
                theta_pi_med = np.median(thetas_pi)
                
            err_mic = abs(theta_mic_med - theta_true)
            err_pi = abs(theta_pi_med - theta_true)
            
            print(f"{sjr:5.1f}\tMic={theta_mic_med:6.2f}(E:{err_mic:5.2f})\tPI-GS={theta_pi_med:6.2f}(E:{err_pi:5.2f})")
            f.write(f"{sjr:.1f}\t{err_mic:.3f}\t{err_pi:.3f}\n")
            f.flush()

    print(f"Data saved to {out_dat}")

if __name__ == '__main__':
    main()

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return sr, data

def main():
    root = "/home/sbplab/jiawei/0222-block/0223-block/"
    f_ldv = os.path.join(root, "0223-LDV-40-boy(+0.4m)-13-block.wav")
    f_micL = os.path.join(root, "0223-MIC-LEFT-40-boy(+0.4m)-13-block.wav")
    f_micR = os.path.join(root, "0223-MIC-RIGHT-40-boy(+0.4m)-13-block.wav")
    
    fs, ldv = load_wav(f_ldv)
    _, micL = load_wav(f_micL)
    _, micR = load_wav(f_micR)
    
    # Use 5 seconds
    n_samples = int(5.0 * fs)
    ldv = ldv[:n_samples]
    micL = micL[:n_samples]
    micR = micR[:n_samples]
    
    nperseg = 2048
    noverlap = 1536
    
    f, t, Z_ldv = stft(ldv, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Z_micL = stft(micL, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Z_micR = stft(micR, fs, nperseg=nperseg, noverlap=noverlap)
    
    # Calculate Cross-Spectra
    R_LR = Z_micL * np.conj(Z_micR)
    R_VL = Z_ldv * np.conj(Z_micL)
    R_VR = Z_ldv * np.conj(Z_micR)
    
    P_LR = np.mean(R_LR / (np.abs(R_LR) + 1e-10), axis=1)
    P_VL = np.mean(R_VL / (np.abs(R_VL) + 1e-10), axis=1)
    P_VR = np.mean(R_VR / (np.abs(R_VR) + 1e-10), axis=1)
    
    c = 343.0
    pos_ldv = np.array([0.0, 0.5])
    pos_micL = np.array([-0.7, 2.0])
    pos_micR = np.array([0.7, 2.0])
    
    X_range = np.linspace(-1.0, 1.0, 200)
    
    # We want to display frequencies up to 8000 Hz
    f_mask = f <= 8000
    f_plot = f[f_mask]
    P_LR = P_LR[f_mask]
    P_VL = P_VL[f_mask]
    P_VR = P_VR[f_mask]
    
    img_LR = np.zeros((len(X_range), len(f_plot)))
    img_PI = np.zeros((len(X_range), len(f_plot)))
    
    for i, X in enumerate(X_range):
        pos_wall = np.array([X, 0.5]) # evaluate physical barrier points
        pos_src = np.array([X, 0.0])  # If target was at X
        
        # for Mic-Mic, we map X to the corresponding target delay
        d_L = np.linalg.norm(pos_src - pos_micL)
        d_R = np.linalg.norm(pos_src - pos_micR)
        tau_LR = (d_L - d_R) / c
        
        # for PI-GS, we use the delay differences across the wall
        d_vL = np.linalg.norm(pos_wall - pos_micL) # from wall point
        d_vR = np.linalg.norm(pos_wall - pos_micR)
        tau_VL_wall = d_vL / c
        tau_VR_wall = d_vR / c
        # Wait, the PI-GS equation Eq 10 is:
        # R_VL(tau_VL(p)) + R_VR(tau_VR(p))
        # from a source at p=(X, 0.0), it reaches wall at some point. 
        # But wait! PI-GS sweeps the target coordinate p=(X, 0.0)!
        # The theoretical delay from LDV anchor to Mic is:
        # tau_Vm(p) = tau_m(p) - tau_V(p)
        # where tau_m(p) = d(p, mic_m)/c, tau_V(p) = d(p, ldv)/c
        tau_V = np.linalg.norm(pos_src - pos_ldv) / c
        tau_VL = (d_L / c) - tau_V
        tau_VR = (d_R / c) - tau_V
        
        for k, freq in enumerate(f_plot):
            omega = 2 * np.pi * freq
            # Mic-Mic GCC-PHAT spectral component
            img_LR[i, k] = np.real(P_LR[k] * np.exp(1j * omega * tau_LR))
            
            # LDV-Mic PI-GS spectral component
            s_VL = np.real(P_VL[k] * np.exp(1j * omega * tau_VL))
            s_VR = np.real(P_VR[k] * np.exp(1j * omega * tau_VR))
            img_PI[i, k] = s_VL + s_VR

    # Save downsampled data for pgfplots (e.g. 50x50 grid)
    out_dat_LR = '/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/freq_x_scalogram_MicMic.dat'
    out_dat_PI = '/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/freq_x_scalogram_PIGS.dat'
    
    down_X = 50
    down_F = 50
    
    idx_X = np.linspace(0, len(X_range)-1, down_X, dtype=int)
    idx_F = np.linspace(0, len(f_plot)-1, down_F, dtype=int)
    
    with open(out_dat_LR, 'w') as f_LR, open(out_dat_PI, 'w') as f_PI:
        f_LR.write("X\tFreq\tValue\n")
        f_PI.write("X\tFreq\tValue\n")
        for ix in idx_X:
            for ik in idx_F:
                f_LR.write(f"{X_range[ix]:.3f}\t{f_plot[ik]:.1f}\t{img_LR[ix, ik]:.4f}\n")
                f_PI.write(f"{X_range[ix]:.3f}\t{f_plot[ik]:.1f}\t{img_PI[ix, ik]:.4f}\n")
            f_LR.write("\n")
            f_PI.write("\n")
            
    print(f"Saved downsampled pgfplots data to *.dat")

if __name__ == '__main__':
    main()

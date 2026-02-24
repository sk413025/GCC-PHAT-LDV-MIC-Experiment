import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

sys.path.append('/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/scripts')
import multi_sensor_fusion_doa as msnf

def main():
    root = "/home/sbplab/jiawei/0222-block/0223-block/"
    
    f_ldv = os.path.join(root, "0223-LDV-40-boy(+0.4m)-13-block.wav")
    f_micL = os.path.join(root, "0223-MIC-LEFT-40-boy(+0.4m)-13-block.wav")
    f_micR = os.path.join(root, "0223-MIC-RIGHT-40-boy(+0.4m)-13-block.wav")
    
    f_jamL = os.path.join(root, "0223-unblock-7(high)/0223-MIC-LEFT-40-boy(-0.8m)-20-unblock.wav")
    f_jamR = os.path.join(root, "0223-unblock-7(high)/0223-MIC-RIGHT-40-boy(-0.8m)-20-unblock.wav")

    fs, tgt_ldv = msnf.load_wav(f_ldv)
    _, tgt_micL = msnf.load_wav(f_micL)
    _, tgt_micR = msnf.load_wav(f_micR)
    
    _, jam_micL = msnf.load_wav(f_jamL)
    _, jam_micR = msnf.load_wav(f_jamR)
    
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
    
    # Ground Truths for speaker 19 (+0.4m)
    ground_truths = msnf.compute_all_ground_truths("19", c=c, d=1.4)
    tau1_true = ground_truths["tau1_true_ms"]
    tau2_true = ground_truths["tau2_true_ms"]
    tau3_true = ground_truths["tau3_true_ms"]
    theta_true = ground_truths["theta_true_deg"]
    
    config = msnf.DEFAULT_CONFIG.copy()
    config["gcc_bandpass_low"] = 500.0
    config["gcc_bandpass_high"] = 2000.0
    
    sjr_list = np.linspace(-40, 20, 31)
    
    mae_mic_only = []
    mae_pi_gs = []
    
    print("--- PGFPlots Data: Jammer Resilience Curve (MSNF E_msnf_3) ---")
    print("SJR_dB\tMic-Mic_MAE\tPI-GS_MAE")
    
    out_dat = '/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/jammer_resilience_curve_msnf.dat'
    with open(out_dat, 'w') as f:
        f.write("SJR_dB\tMic-Mic_MAE\tPI-GS_MAE\n")
        
        for sjr in sjr_list:
            scale = (tgt_rms / jam_rms) * (10.0 ** (-sjr / 20.0))
            
            mix_micL = tgt_micL + scale * jam_micL
            mix_micR = tgt_micR + scale * jam_micR
            
            # Using MSNF internal pipeline for a single window
            tau1_ms, tau1_psr = msnf._measure_tau1(mix_micL, mix_micR, fs, config, ground_truths)
            tau2_ms, tau2_psr = msnf._measure_tau_ldv_mic(tgt_ldv, mix_micL, fs, config, tau2_true)
            tau3_ms, tau3_psr = msnf._measure_tau_ldv_mic(tgt_ldv, mix_micR, fs, config, tau3_true)
            
            psr_floor = -20.0
            psr_ceiling = 20.0
            w1 = msnf.psr_to_weight(tau1_psr, psr_floor, psr_ceiling)
            w2 = msnf.psr_to_weight(tau2_psr, psr_floor, psr_ceiling)
            w3 = msnf.psr_to_weight(tau3_psr, psr_floor, psr_ceiling)
            
            # Method A: Mic-Mic
            theta_A = msnf.tau_to_doa(tau1_ms, c, 1.4)
            err_mic = abs(theta_A - theta_true)
            
            # Method E: MSNF-3 (PI-GS)
            def _tau1_m(x): return msnf.tau1_model(x, c)
            def _tau2_m(x): return msnf.tau2_model(x, c)
            def _tau3_m(x): return msnf.tau3_model(x, c)
            
            x_E, theta_E = msnf.solve_msnf(
                [(tau1_ms / 1000.0, _tau1_m),
                 (tau2_ms / 1000.0, _tau2_m),
                 (tau3_ms / 1000.0, _tau3_m)],
                [w1, w2, w3],
                c,
                1.4
            )
            err_pi = abs(theta_E - theta_true)
            
            mae_mic_only.append(err_mic)
            mae_pi_gs.append(err_pi)
            
            print(f"{sjr:.1f}\t{err_mic:.3f}\t{err_pi:.3f}")
            f.write(f"{sjr:.1f}\t{err_mic:.3f}\t{err_pi:.3f}\n")
            f.flush()

    print(f"Data saved to {out_dat}")
    
    # Check what happens if we use tau_expected_ms = 0 (completely unguided) to see if it locks onto jammer
    # The default config sets tau2_guided_radius_ms = 2.0 (guided search!)
    # That means the LDV is literally being given the ground truth to look around! 
    # This is why it always succeeds!
    
if __name__ == '__main__':
    main()

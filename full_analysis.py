#!/usr/bin/env python3
"""
GCC-PHAT 完整實驗分析
======================
涵蓋所有資料夾：03~07 (音量敏感度) + 18~22 (LDV vs MIC) + 23~24 (chirp)

使用方式:
    cd /home/sbplab/jiawei/data
    python3 full_analysis.py
"""

import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import butter, filtfilt
import os
import sys

# ============================================================
# 環境參數
# ============================================================
Y_DISTANCE = 2.0          # 垂直距離 (m)
MIC_SPACING = 1.4         # 麥克風間距 (m)
MIC1_X = -MIC_SPACING / 2 # 左麥克風 X (-0.7m)
MIC2_X = +MIC_SPACING / 2 # 右麥克風 X (+0.7m)
LDV_X = 0.0              # LDV X
LDV_Y = 0.5              # LDV Y
SOUND_SPEED = 343         # m/s
MAX_TAU = 0.01            # 最大搜索延遲 (s)
FS = 48000                # 取樣率

# Speaker 位置配置
SPEAKER_POSITIONS = {
    '03-0.3V':  0.8,  '04-0.1V':  0.8,  '05-0.01V':  0.8,
    '07-0.005V': 0.8, '06-0.001V': 0.8,
    '18-0.1V':  0.8,  '19-0.1V':  0.4,  '20-0.1V':  0.0,
    '21-0.1V': -0.4,  '22-0.1V': -0.8,
    '23-chirp(-0.8m)': -0.8,  '24-chirp(-0.4m)': -0.4,
}

# ============================================================
# 階段一：音量敏感度 (MIC-MIC only, chirp)
# ============================================================
PHASE1_FOLDERS = [
    ('03-0.3V',   '0128-LEFT-MIC-3.wav',  '0128-RIGHT-MIC-3.wav',  0.3),
    ('04-0.1V',   '0128-LEFT-MIC-4.wav',  '0128-RIGHT-MIC-4.wav',  0.1),
    ('05-0.01V',  '0128-LEFT-MIC-5.wav',  '0128-RIGHT-MIC-5.wav',  0.01),
    ('07-0.005V', '0128-LEFT-MIC-7.wav',  '0128-RIGHT-MIC-7.wav',  0.005),
    ('06-0.001V', '0128-LEFT-MIC-6.wav',  '0128-RIGHT-MIC-6.wav',  0.001),
]

# ============================================================
# 階段二：LDV vs MIC (0.1V boy 語音)
# ============================================================
PHASE2_FOLDERS = [
    ('18-0.1V', '0128-LDV-18-boy-320.wav', '0128-LEFT-MIC-18-boy-320.wav', '0128-RIGHT-MIC-18-boy-320.wav'),
    ('19-0.1V', '0128-LDV-19-boy-320.wav', '0128-LEFT-MIC-19-boy-320.wav', '0128-RIGHT-MIC-19-boy-320.wav'),
    ('20-0.1V', '0128-LDV-20-boy-320.wav', '0128-LEFT-MIC-20-boy-320.wav', '0128-RIGHT-MIC-20-boy-320.wav'),
    ('21-0.1V', '0128-LDV-21-boy-320.wav', '0128-LEFT-MIC-21-boy-320.wav', '0128-RIGHT-MIC-21-boy-320.wav'),
    ('22-0.1V', '0128-LDV-22-boy-320.wav', '0128-LEFT-MIC-22-boy-320.wav', '0128-RIGHT-MIC-22-boy-320.wav'),
]

# ============================================================
# 階段三：chirp 校準
# ============================================================
PHASE3_FOLDERS = [
    ('23-chirp(-0.8m)', '0128-LDV-23-chirp(-0.8m).wav', '0128-LEFT-MIC-23-chirp(-0.8m).wav', '0128-RIGHT-MIC-23-chirp(-0.8m).wav'),
    ('24-chirp(-0.4m)', '0128-LDV-24-chirp(-0.4m).wav', '0128-LEFT-MIC-24-chirp(-0.4m).wav', '0128-RIGHT-MIC-24-chirp(-0.4m).wav'),
]


# ============================================================
# 帶通濾波
# ============================================================
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


# ============================================================
# GCC-PHAT (含拋物線插值 + 品質指標)
# ============================================================
def gcc_phat(sig1, sig2, fs, max_tau=None):
    """
    Returns: tau, peak, par, psr_db, cc
    """
    n = len(sig1) + len(sig2)
    SIG1 = fft(sig1, n)
    SIG2 = fft(sig2, n)
    R = SIG1 * np.conj(SIG2)
    R = R / (np.abs(R) + 1e-10)
    cc = np.real(ifft(R))

    if max_tau:
        max_shift = int(max_tau * fs)
    else:
        max_shift = n // 2

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    abs_cc = np.abs(cc)

    # 粗略峰值
    peak_idx = np.argmax(abs_cc)
    peak_val = abs_cc[peak_idx]

    # 拋物線插值
    if 1 <= peak_idx <= len(cc) - 2:
        y0 = abs_cc[peak_idx - 1]
        y1 = abs_cc[peak_idx]
        y2 = abs_cc[peak_idx + 1]
        denom = 2 * (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            delta = (y0 - y2) / denom
        else:
            delta = 0.0
        refined_idx = peak_idx + delta
    else:
        refined_idx = float(peak_idx)

    shift = refined_idx - max_shift
    tau = shift / fs

    # PAR
    avg_abs = np.mean(abs_cc)
    par = peak_val / avg_abs if avg_abs > 0 else 0

    # PSR: 排除主峰 ±50 樣本
    exclude = 50
    sidelobe_mask = np.ones(len(abs_cc), dtype=bool)
    lo = max(0, peak_idx - exclude)
    hi = min(len(abs_cc), peak_idx + exclude + 1)
    sidelobe_mask[lo:hi] = False
    sidelobes = abs_cc[sidelobe_mask]
    if len(sidelobes) > 0 and np.max(sidelobes) > 0:
        psr_db = 20 * np.log10(peak_val / np.max(sidelobes))
    else:
        psr_db = float('inf')

    return tau, peak_val, par, psr_db, cc


# ============================================================
# 理論計算
# ============================================================
def calc_theory(speaker_x):
    """計算所有理論 TDOA"""
    d_left = np.sqrt((speaker_x - MIC1_X)**2 + Y_DISTANCE**2)
    d_right = np.sqrt((speaker_x - MIC2_X)**2 + Y_DISTANCE**2)
    d_ldv = np.sqrt((speaker_x - LDV_X)**2 + (0 - LDV_Y)**2)

    t_left = d_left / SOUND_SPEED
    t_right = d_right / SOUND_SPEED
    t_ldv = d_ldv / SOUND_SPEED

    return {
        'mic_mic': (t_left - t_right),        # LEFT vs RIGHT
        'ldv_left': (t_ldv - t_left),          # LDV vs LEFT
        'ldv_right': (t_ldv - t_right),        # LDV vs RIGHT
        'd_left': d_left, 'd_right': d_right, 'd_ldv': d_ldv,
    }


def load_wav(path):
    fs, audio = wavfile.read(path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    return fs, audio.astype(np.float64)


# ============================================================
# 階段一：音量敏感度
# ============================================================
def run_phase1():
    print("=" * 100)
    print("階段一：音量敏感度測試 (03~07, Speaker @ +0.8m, chirp, MIC-MIC)")
    print("=" * 100)

    theory = calc_theory(0.8)
    theory_tau = theory['mic_mic']
    theory_path = theory_tau * SOUND_SPEED
    print(f"理論 MIC-MIC τ = {theory_tau*1000:.4f} ms, 路徑差 = {theory_path*100:.2f} cm")
    print()
    print(f"{'資料夾':<15} | {'音量(V)':>8} | {'τ(ms)':>10} | {'Peak':>8} | {'PAR':>8} | {'PSR(dB)':>8} | {'路徑差(cm)':>10} | {'誤差(cm)':>9} | {'誤差%':>8} | 狀態")
    print("-" * 115)

    results = []
    for folder, left_f, right_f, voltage in PHASE1_FOLDERS:
        lp = os.path.join(folder, left_f)
        rp = os.path.join(folder, right_f)
        if not os.path.exists(lp):
            print(f"{folder:<15} | 檔案不存在")
            continue

        _, left = load_wav(lp)
        _, right = load_wav(rp)
        min_len = min(len(left), len(right))
        left, right = left[:min_len], right[:min_len]

        tau, peak, par, psr, _ = gcc_phat(left, right, FS, MAX_TAU)
        path_diff = tau * SOUND_SPEED
        err_cm = (path_diff - theory_path) * 100
        err_pct = err_cm / (theory_path * 100) * 100 if theory_path != 0 else float('inf')
        status = "✓" if abs(err_pct) < 10 else "❌"

        print(f"{folder:<15} | {voltage:>8.4f} | {tau*1000:>10.4f} | {peak:>8.4f} | {par:>8.1f} | {psr:>8.2f} | {path_diff*100:>10.2f} | {err_cm:>+9.2f} | {err_pct:>+7.2f}% | {status}")
        results.append({
            'folder': folder, 'voltage': voltage, 'tau': tau, 'peak': peak,
            'par': par, 'psr': psr, 'err_pct': err_pct, 'status': status
        })

    print()
    return results


# ============================================================
# 階段二：LDV vs MIC 比較
# ============================================================
def run_phase2():
    print("=" * 100)
    print("階段二：LDV vs MIC 比較 (18~22, 0.1V boy 語音, 500-2000Hz 帶通, 100-600s)")
    print("=" * 100)
    print()

    all_results = []

    for folder, ldv_f, left_f, right_f in PHASE2_FOLDERS:
        speaker_x = SPEAKER_POSITIONS[folder]
        theory = calc_theory(speaker_x)

        lp = os.path.join(folder, left_f)
        rp = os.path.join(folder, right_f)
        vp = os.path.join(folder, ldv_f)

        if not os.path.exists(lp):
            print(f"{folder}: 檔案不存在")
            continue

        print(f"--- {folder} (Speaker @ {speaker_x:+.1f}m) ---")
        print(f"  理論: MIC-MIC={theory['mic_mic']*1000:.4f}ms, LDV-LEFT={theory['ldv_left']*1000:.4f}ms, LDV-RIGHT={theory['ldv_right']*1000:.4f}ms")

        _, left_raw = load_wav(lp)
        _, right_raw = load_wav(rp)
        _, ldv_raw = load_wav(vp)

        # 截取 100-600s
        start = 100 * FS
        end = 600 * FS
        min_len = min(len(left_raw), len(right_raw), len(ldv_raw))
        if min_len < end:
            end = min_len
            print(f"  警告: 錄音長度不足 600s, 使用到 {end/FS:.1f}s")

        left_seg = left_raw[start:end]
        right_seg = right_raw[start:end]
        ldv_seg = ldv_raw[start:end]

        # 帶通濾波
        left_bp = bandpass_filter(left_seg, 500, 2000, FS)
        right_bp = bandpass_filter(right_seg, 500, 2000, FS)
        ldv_bp = bandpass_filter(ldv_seg, 500, 2000, FS)

        # 三種配對
        pairs = [
            ('MIC-MIC',    left_bp, right_bp, theory['mic_mic']),
            ('LDV-LEFT',   ldv_bp,  left_bp,  theory['ldv_left']),
            ('LDV-RIGHT',  ldv_bp,  right_bp, theory['ldv_right']),
        ]

        print(f"  {'配對':<12} | {'τ(ms)':>10} | {'理論τ(ms)':>10} | {'Peak':>8} | {'PAR':>8} | {'PSR(dB)':>8}")
        print(f"  {'-'*70}")

        folder_results = {'folder': folder, 'speaker_x': speaker_x}
        for pair_name, s1, s2, theory_tau in pairs:
            tau, peak, par, psr, _ = gcc_phat(s1, s2, FS, MAX_TAU)
            print(f"  {pair_name:<12} | {tau*1000:>10.4f} | {theory_tau*1000:>10.4f} | {peak:>8.4f} | {par:>8.1f} | {psr:>8.2f}")
            folder_results[pair_name] = {
                'tau': tau, 'peak': peak, 'par': par, 'psr': psr,
                'theory_tau': theory_tau
            }

        # Δτ 分析 (命名判定)
        if 'LDV-LEFT' in folder_results and 'LDV-RIGHT' in folder_results:
            dt_meas = folder_results['LDV-LEFT']['tau'] - folder_results['LDV-RIGHT']['tau']
            dt_theory = theory['ldv_left'] - theory['ldv_right']
            sign_match = "✅" if (dt_meas * dt_theory > 0 or abs(dt_theory) < 1e-6) else "❌"
            print(f"  Δτ(LDV-L - LDV-R): 測量={dt_meas*1000:.4f}ms, 理論={dt_theory*1000:.4f}ms, 符號{'一致' if sign_match == '✅' else '不一致'} {sign_match}")
            folder_results['delta_tau_meas'] = dt_meas
            folder_results['delta_tau_theory'] = dt_theory
            folder_results['sign_match'] = sign_match

        # LDV 延遲估算
        for pn in ['LDV-LEFT', 'LDV-RIGHT']:
            if pn in folder_results:
                delay = folder_results[pn]['tau'] - folder_results[pn]['theory_tau']
                folder_results[pn]['ldv_delay'] = delay

        all_results.append(folder_results)
        print()

    return all_results


# ============================================================
# 階段三：chirp 校準
# ============================================================
def run_phase3():
    print("=" * 100)
    print("階段三：Chirp 校準 (23~24, LDV + MIC)")
    print("=" * 100)
    print()

    all_results = []

    for folder, ldv_f, left_f, right_f in PHASE3_FOLDERS:
        speaker_x = SPEAKER_POSITIONS[folder]
        theory = calc_theory(speaker_x)

        lp = os.path.join(folder, left_f)
        rp = os.path.join(folder, right_f)
        vp = os.path.join(folder, ldv_f)

        if not os.path.exists(lp):
            print(f"{folder}: 檔案不存在")
            continue

        print(f"--- {folder} (Speaker @ {speaker_x:+.1f}m) ---")

        _, left_raw = load_wav(lp)
        _, right_raw = load_wav(rp)
        _, ldv_raw = load_wav(vp)

        min_len = min(len(left_raw), len(right_raw), len(ldv_raw))
        left_raw = left_raw[:min_len]
        right_raw = right_raw[:min_len]
        ldv_raw = ldv_raw[:min_len]

        duration = min_len / FS
        print(f"  錄音長度: {duration:.2f}s")

        # chirp 用全段，不做帶通濾波（chirp 本身就是寬頻訊號）
        pairs = [
            ('MIC-MIC',   left_raw,  right_raw, theory['mic_mic']),
            ('LDV-LEFT',  ldv_raw,   left_raw,  theory['ldv_left']),
            ('LDV-RIGHT', ldv_raw,   right_raw, theory['ldv_right']),
        ]

        print(f"  {'配對':<12} | {'τ(ms)':>10} | {'理論τ(ms)':>10} | {'Peak':>8} | {'PAR':>8} | {'PSR(dB)':>8}")
        print(f"  {'-'*70}")

        folder_results = {'folder': folder, 'speaker_x': speaker_x}
        for pair_name, s1, s2, theory_tau in pairs:
            tau, peak, par, psr, _ = gcc_phat(s1, s2, FS, MAX_TAU)
            print(f"  {pair_name:<12} | {tau*1000:>10.4f} | {theory_tau*1000:>10.4f} | {peak:>8.4f} | {par:>8.1f} | {psr:>8.2f}")
            folder_results[pair_name] = {
                'tau': tau, 'peak': peak, 'par': par, 'psr': psr,
                'theory_tau': theory_tau
            }

        # LDV 延遲估算
        for pn in ['LDV-LEFT', 'LDV-RIGHT']:
            if pn in folder_results:
                delay = folder_results[pn]['tau'] - folder_results[pn]['theory_tau']
                folder_results[pn]['ldv_delay'] = delay

        all_results.append(folder_results)
        print()

    return all_results


# ============================================================
# 彙總
# ============================================================
def print_summary(p1, p2, p3):
    print("=" * 100)
    print("完整彙總")
    print("=" * 100)

    # 階段一彙總
    print("\n【階段一：音量敏感度】")
    success = [r for r in p1 if r['status'] == '✓']
    fail = [r for r in p1 if r['status'] == '❌']
    print(f"  成功: {len(success)}/{len(p1)}")
    if success:
        vmin = min(r['voltage'] for r in success)
        vmax = max(r['voltage'] for r in success)
        print(f"  有效範圍: {vmin}V ~ {vmax}V ({vmax/vmin:.0f}x)")
        avg_err = np.mean([abs(r['err_pct']) for r in success])
        print(f"  平均誤差: {avg_err:.2f}%")

    # 階段二：LDV vs MIC 品質比較
    print("\n【階段二：LDV vs MIC 品質比較 (18~22)】")
    print(f"  {'資料夾':<12} | {'Spk位置':>6} | {'MIC Peak':>8} | {'LDV-L Peak':>10} | {'LDV-R Peak':>10} | {'MIC PAR':>7} | {'LDV-L PAR':>9} | {'LDV-R PAR':>9} | {'命名':>4}")
    print(f"  {'-'*105}")
    for r in p2:
        mm = r.get('MIC-MIC', {})
        ll = r.get('LDV-LEFT', {})
        lr = r.get('LDV-RIGHT', {})
        sm = r.get('sign_match', '?')
        print(f"  {r['folder']:<12} | {r['speaker_x']:>+5.1f}m | {mm.get('peak',0):>8.4f} | {ll.get('peak',0):>10.4f} | {lr.get('peak',0):>10.4f} | {mm.get('par',0):>7.1f} | {ll.get('par',0):>9.1f} | {lr.get('par',0):>9.1f} | {sm}")

    # LDV 延遲估算
    print("\n【LDV 設備延遲估算】")
    delays = []
    for r in p2 + p3:
        for pn in ['LDV-LEFT', 'LDV-RIGHT']:
            if pn in r and 'ldv_delay' in r[pn]:
                d = r[pn]['ldv_delay']
                delays.append(d)
                print(f"  {r['folder']} {pn}: {d*1000:.4f} ms")
    if delays:
        print(f"  平均延遲: {np.mean(delays)*1000:.4f} ms")
        print(f"  範圍: {np.min(delays)*1000:.4f} ~ {np.max(delays)*1000:.4f} ms")

    # 階段三
    print("\n【階段三：Chirp 校準】")
    for r in p3:
        mm = r.get('MIC-MIC', {})
        print(f"  {r['folder']}: MIC-MIC τ={mm.get('tau',0)*1000:.4f}ms (理論={mm.get('theory_tau',0)*1000:.4f}ms), Peak={mm.get('peak',0):.4f}")


def main():
    os.chdir('/home/sbplab/jiawei/data')
    print("GCC-PHAT 完整實驗分析")
    print(f"工作目錄: {os.getcwd()}")
    print()

    p1 = run_phase1()
    p2 = run_phase2()
    p3 = run_phase3()
    print_summary(p1, p2, p3)

    return p1, p2, p3


if __name__ == "__main__":
    main()

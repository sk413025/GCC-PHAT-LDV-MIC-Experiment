"""C1 — Apply bandpass filter to all channels before GCC.

Test multiple bands to learn which works for which signal type.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scipy.signal import butter, filtfilt
from _strategy_runner import run_strategy, print_strategy, default_preprocessor


def make_bandpass(lo, hi, order=4):
    def pp(chans, sr):
        chans = default_preprocessor(chans, sr)
        b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
        return {ch: filtfilt(b, a, x) for ch, x in chans.items()}
    return pp


def main():
    bands = [
        (200, 4000, "C1a_bp200_4000"),
        (500, 2000, "C1b_bp500_2000"),
        (300, 1500, "C1c_bp300_1500"),
        (1000, 4000, "C1d_bp1000_4000"),
    ]
    for lo, hi, sid in bands:
        s = run_strategy(sid, preprocessor=make_bandpass(lo, hi),
                        pigs_kwargs=dict(band_hz=(lo, hi)),
                        micmic_kwargs=dict(band_hz=(lo, hi)))
        print_strategy(s)


if __name__ == "__main__":
    main()

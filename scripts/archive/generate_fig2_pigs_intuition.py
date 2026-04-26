#!/usr/bin/env python3
"""
Generate Figure 2: PI-GS geometric intuition — three-panel visualization.

Panel (a): τ_VL(p) pure geometry field — iso-delay contours (the "terrain map")
Panel (b): Two single-channel likelihood heatmaps R_VL(τ_VL(p)) and R_VR(τ_VR(p))
           shown as colored ridges overlaid on geometry
Panel (c): Joint score S(p) = |R_VL(.)| + |R_VR(.)| heatmap with peak at p*

Uses the real testbed geometry from the paper.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ── Geometry (from paper / scripts) ──────────────────────────────────
V  = np.array([0.0, 0.5])     # LDV point
ML = np.array([-0.7, 2.0])    # Mic Left
MR = np.array([0.7, 2.0])     # Mic Right
C  = 343.0                     # speed of sound m/s

# Source behind barrier (X varies, Y ≈ 0 for source on barrier surface)
# Search region Ω: source side
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -0.8, 0.0     # behind the barrier (source side)

# True source position
P_TRUE = np.array([0.4, -0.25])

# ── Grid ─────────────────────────────────────────────────────────────
NX, NY = 400, 300
xs = np.linspace(X_MIN, X_MAX, NX)
ys = np.linspace(Y_MIN, Y_MAX, NY)
XX, YY = np.meshgrid(xs, ys)

def dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def dist_grid(XX, YY, sensor):
    return np.sqrt((XX - sensor[0])**2 + (YY - sensor[1])**2)

# ── Delay fields (relative: mic - LDV) ──────────────────────────────
dV  = dist_grid(XX, YY, V)
dML = dist_grid(XX, YY, ML)
dMR = dist_grid(XX, YY, MR)

tau_VL = (dML - dV) / C * 1000  # ms
tau_VR = (dMR - dV) / C * 1000  # ms

# True delays
tau_VL_true = (dist(P_TRUE, ML) - dist(P_TRUE, V)) / C * 1000
tau_VR_true = (dist(P_TRUE, MR) - dist(P_TRUE, V)) / C * 1000

# ── Synthetic GCC-PHAT-like correlation (1D function of tau) ─────────
def synthetic_gcc(tau, tau_true, width=0.08, noise_peaks=None):
    """
    Simulate a GCC-PHAT-like correlation function:
    - Main peak at tau_true
    - Optional spurious peaks
    - Background noise floor
    """
    # Main peak (narrow, dominant)
    R = np.exp(-0.5 * ((tau - tau_true) / width)**2)

    # Spurious / multipath peaks
    if noise_peaks is not None:
        for (tp, amp, w) in noise_peaks:
            R += amp * np.exp(-0.5 * ((tau - tp) / w)**2)

    # Small noise floor
    rng = np.random.default_rng(42)
    noise = 0.05 * np.abs(rng.standard_normal(tau.shape)) if isinstance(tau, np.ndarray) else 0
    R = np.abs(R) + noise
    return R

# Spurious peaks for VL channel
spurious_VL = [
    (tau_VL_true + 0.7, 0.35, 0.12),   # multipath echo
    (tau_VL_true - 1.0, 0.20, 0.15),   # another spurious
]
spurious_VR = [
    (tau_VR_true - 0.5, 0.30, 0.10),
    (tau_VR_true + 1.2, 0.25, 0.14),
]

# Evaluate correlation on the 2D grid (map tau field → score)
R_VL_map = synthetic_gcc(tau_VL, tau_VL_true, width=0.08, noise_peaks=spurious_VL)
R_VR_map = synthetic_gcc(tau_VR, tau_VR_true, width=0.08, noise_peaks=spurious_VR)

# Normalize each to [0, 1]
R_VL_map /= R_VL_map.max()
R_VR_map /= R_VR_map.max()

# Joint score
S_map = R_VL_map + R_VR_map
S_map /= S_map.max()

# ── Figure setup ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), dpi=300)
plt.subplots_adjust(wspace=0.08, left=0.06, right=0.97, top=0.88, bottom=0.15)

# Custom colormaps
cmap_blue = LinearSegmentedColormap.from_list("blue_ridge",
    ["#0a0a2e", "#0d1b5e", "#1a3c8f", "#2d6abf", "#4da6ff", "#8cd3ff", "#d0eeff", "#ffffff"], N=256)
cmap_red = LinearSegmentedColormap.from_list("red_ridge",
    ["#2a0a0a", "#5e0d0d", "#8f1a1a", "#bf3030", "#ff4d4d", "#ff8c8c", "#ffd0d0", "#ffffff"], N=256)
cmap_joint = LinearSegmentedColormap.from_list("joint_hot",
    ["#0a0a1a", "#1a0a3c", "#3c0a5e", "#6b1a8f", "#a040bf", "#d070e0", "#ff8cff",
     "#ffcc66", "#ffee44", "#ffffff"], N=256)

# Shared appearance
sensor_kw = dict(zorder=10, edgecolors="white", linewidths=0.6, s=28)
outline = [pe.withStroke(linewidth=2, foreground="black")]
label_kw = dict(fontsize=6, fontweight="bold", zorder=11,
                path_effects=outline, ha="center")

def add_sensors(ax):
    """Draw V, L, R sensor markers and barrier line."""
    # Barrier at y=0
    ax.axhline(0, color="white", linewidth=1.2, linestyle="-", alpha=0.7)
    ax.text(X_MAX - 0.05, 0.03, "barrier", fontsize=5, color="white",
            alpha=0.7, ha="right", va="bottom",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    # Sensors (on receiver side, y > 0) — draw as arrows pointing in
    ax.scatter(*V, marker="^", color="#00ff88", **sensor_kw)
    ax.text(V[0], V[1] + 0.06, "$V$", color="#00ff88", **label_kw)

    ax.scatter(*ML, marker="s", color="#66aaff", **sensor_kw)
    ax.text(ML[0], ML[1] + 0.06, "$L$", color="#66aaff", **label_kw)

    ax.scatter(*MR, marker="s", color="#ff6666", **sensor_kw)
    ax.text(MR[0], MR[1] + 0.06, "$R$", color="#ff6666", **label_kw)

def add_true_source(ax, with_label=True):
    ax.scatter(*P_TRUE, marker="*", color="#ffff00", s=60, zorder=11,
               edgecolors="black", linewidths=0.4)
    if with_label:
        ax.text(P_TRUE[0] + 0.08, P_TRUE[1] - 0.06, r"$\mathbf{p}^\star$",
                fontsize=7, color="#ffff00", fontweight="bold",
                path_effects=outline, va="top")

def format_ax(ax, title, show_ylabel=False):
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX + 0.05)  # tiny margin at top for barrier label
    ax.set_aspect("equal")
    ax.set_xlabel("$X$ (m)", fontsize=6, labelpad=2)
    if show_ylabel:
        ax.set_ylabel("$Y$ (m)", fontsize=6, labelpad=2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=5, length=2)
    ax.set_title(title, fontsize=7, pad=4)

# ═══════════════════════════════════════════════════════════════════════
# Panel (a): Pure geometry — iso-delay contours of τ_VL and τ_VR
# ═══════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_facecolor("#0a0a2e")

# τ_VL contours (blue)
n_contours = 12
levels_VL = np.linspace(tau_VL.min(), tau_VL.max(), n_contours)
cs_vl = ax.contour(XX, YY, tau_VL, levels=levels_VL, colors="#4da6ff",
                    linewidths=0.6, alpha=0.7)

# τ_VR contours (red)
levels_VR = np.linspace(tau_VR.min(), tau_VR.max(), n_contours)
cs_vr = ax.contour(XX, YY, tau_VR, levels=levels_VR, colors="#ff6666",
                    linewidths=0.6, alpha=0.7)

# Highlight the TRUE delay contours (thicker)
ax.contour(XX, YY, tau_VL, levels=[tau_VL_true], colors="#4da6ff",
           linewidths=2.0, alpha=1.0)
ax.contour(XX, YY, tau_VR, levels=[tau_VR_true], colors="#ff6666",
           linewidths=2.0, alpha=1.0)

# Labels for highlighted contours
# Find a point on the H_L contour for labeling
mask_hl = np.abs(tau_VL - tau_VL_true) < 0.02
iy_hl, ix_hl = np.where(mask_hl)
if len(ix_hl) > 0:
    idx = len(ix_hl) // 5
    ax.text(xs[ix_hl[idx]] - 0.12, ys[iy_hl[idx]], "$H_L$", fontsize=6.5,
            color="#4da6ff", fontweight="bold", path_effects=outline)

mask_hr = np.abs(tau_VR - tau_VR_true) < 0.02
iy_hr, ix_hr = np.where(mask_hr)
if len(ix_hr) > 0:
    idx = len(ix_hr) * 4 // 5
    ax.text(xs[ix_hr[idx]] + 0.06, ys[iy_hr[idx]], "$H_R$", fontsize=6.5,
            color="#ff6666", fontweight="bold", path_effects=outline)

add_sensors(ax)
add_true_source(ax)
format_ax(ax, r"(a) Geometry: $\tau_{VL}(\mathbf{p})$, $\tau_{VR}(\mathbf{p})$",
          show_ylabel=True)

# Legend-like annotation
ax.text(X_MIN + 0.05, Y_MIN + 0.05, "iso-delay\ncontours", fontsize=4.5,
        color="white", alpha=0.6, va="bottom",
        path_effects=[pe.withStroke(linewidth=1, foreground="black")])

# ═══════════════════════════════════════════════════════════════════════
# Panel (b): Single-channel heatmaps overlaid (blue=VL, red=VR ridges)
# ═══════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_facecolor("#0a0a1a")

# Blend two heatmaps: use alpha compositing
# Blue channel for VL, Red channel for VR
rgb = np.zeros((*R_VL_map.shape, 3))

# VL → blue tones
rgb[:, :, 0] += R_VL_map * 0.15   # slight red tint
rgb[:, :, 1] += R_VL_map * 0.45   # some green
rgb[:, :, 2] += R_VL_map * 1.0    # full blue

# VR → red tones
rgb[:, :, 0] += R_VR_map * 1.0    # full red
rgb[:, :, 1] += R_VR_map * 0.25   # slight green
rgb[:, :, 2] += R_VR_map * 0.15   # slight blue

# Normalize
rgb = np.clip(rgb, 0, 1)
# Boost contrast
rgb = rgb ** 0.65

ax.imshow(rgb, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin="lower",
          aspect="auto", interpolation="bilinear")

# Overlay the true-delay contour lines
ax.contour(XX, YY, tau_VL, levels=[tau_VL_true], colors="#88ccff",
           linewidths=1.0, alpha=0.8, linestyles="--")
ax.contour(XX, YY, tau_VR, levels=[tau_VR_true], colors="#ffaaaa",
           linewidths=1.0, alpha=0.8, linestyles="--")

# Label ridges
# Find bright VL ridge point
peak_vl_iy, peak_vl_ix = np.unravel_index(R_VL_map.argmax(), R_VL_map.shape)
ax.annotate(r"$R_{VL}$ ridge", xy=(xs[peak_vl_ix], ys[peak_vl_iy]),
            xytext=(-0.7, -0.15), fontsize=5, color="#88ccff",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#88ccff", lw=0.6),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

peak_vr_iy, peak_vr_ix = np.unravel_index(R_VR_map.argmax(), R_VR_map.shape)
ax.annotate(r"$R_{VR}$ ridge", xy=(xs[peak_vr_ix], ys[peak_vr_iy]),
            xytext=(0.6, -0.65), fontsize=5, color="#ffaaaa",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#ffaaaa", lw=0.6),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

add_sensors(ax)
add_true_source(ax)
format_ax(ax, r"(b) Single-pair scores: $|R_{VL}(\cdot)|$, $|R_{VR}(\cdot)|$")

# ═══════════════════════════════════════════════════════════════════════
# Panel (c): Joint S(p) heatmap — the "treasure map"
# ═══════════════════════════════════════════════════════════════════════
ax = axes[2]
ax.set_facecolor("#0a0a1a")

im = ax.imshow(S_map, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin="lower",
               aspect="auto", cmap=cmap_joint, vmin=0, vmax=1,
               interpolation="bilinear")

# Contour lines of S(p)
levels_S = [0.5, 0.65, 0.8, 0.9, 0.95]
ax.contour(XX, YY, S_map, levels=levels_S, colors="white",
           linewidths=0.4, alpha=0.5)

# Mark the argmax
peak_iy, peak_ix = np.unravel_index(S_map.argmax(), S_map.shape)
p_hat = np.array([xs[peak_ix], ys[peak_iy]])
ax.scatter(*p_hat, marker="x", color="white", s=40, linewidths=1.5, zorder=12)
ax.text(p_hat[0] + 0.1, p_hat[1] + 0.04, r"$\hat{\mathbf{p}}$",
        fontsize=7, color="white", fontweight="bold",
        path_effects=outline)

# Annotate the joint max
ax.annotate("ridges\nintersect", xy=(p_hat[0], p_hat[1]),
            xytext=(-0.5, -0.6), fontsize=5, color="white",
            fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

# Mini colorbar
cax = fig.add_axes([0.88, 0.18, 0.008, 0.35])
cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
cb.ax.tick_params(labelsize=4, length=1.5)
cb.set_label("$S(\\mathbf{p})$", fontsize=5, labelpad=2)

add_sensors(ax)
add_true_source(ax, with_label=False)
format_ax(ax, r"(c) Joint: $S(\mathbf{p}) = |R_{VL}| + |R_{VR}|$")

# ── Save ─────────────────────────────────────────────────────────────
out_path = "paper/fig2_pigs_intuition.pdf"
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
print(f"Saved → {out_path}")

# Also save PNG for quick preview
out_png = "paper/fig2_pigs_intuition.png"
fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03, dpi=300)
print(f"Saved → {out_png}")

plt.close()

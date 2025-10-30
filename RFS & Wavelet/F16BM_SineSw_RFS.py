import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft, rfftfreq

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ----------- USER CONFIGURATION -----------
# Plot configuration
levels_to_compute = [1, 3, 5, 7]  # Levels to compute RFS
location_i = 2  # DOF i (location where acceleration is measured)
location_j = 3  # DOF j (location across the nonlinear connection)

# Data parameters
fs = 400              # Sampling frequency in Hz
dt = 1 / fs           # Time step

# Mode isolation parameters
target_freq = 7.3     # Target mode frequency [Hz]
freq_low = 6.5        # Lower frequency bound [Hz]
freq_high = 8.2       # Upper frequency bound [Hz]

# Thresholds 
vel_thresh_frac = 0.05     # % of max |v_rel|
disp_thresh_frac = 0.05    # % of max |x_rel|

# Minimum floors in case of tiny signals
vel_thresh_min = 1e-6
disp_thresh_min = 1e-7

# Effective mass (normalized).
m_eff = 1.0

# Input CSVs
data_dict = {
    1: pd.read_csv('BenchmarkData/F16Data_SineSw_Level1.csv'),
    3: pd.read_csv('BenchmarkData/F16Data_SineSw_Level3.csv'),
    5: pd.read_csv('BenchmarkData/F16Data_SineSw_Level5.csv'),
    7: pd.read_csv('BenchmarkData/F16Data_SineSw_Level7.csv')
}
# ------------------------------------------

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass Butterworth filter to isolate a specific frequency band.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, signal)

def band_limited_integrate_fft(a_t, fs, f_lo, f_hi, order=1):
    """
    Band-limited integration via FFT division by (jω)^order inside [f_lo, f_hi].
    - a_t: array (time), already zero-mean recommended
    - order: 1 -> velocity, 2 -> displacement; 0 -> band-pass only (no integration)
    Returns time-domain signal integrated within the band; zeros outside.
    """
    n = len(a_t)
    A = rfft(a_t - np.mean(a_t))
    f = rfftfreq(n, 1.0/fs)
    mask = (f >= f_lo) & (f <= f_hi)
    Y = np.zeros_like(A, dtype=np.complex128)
    if order == 0:
        Y[mask] = A[mask]
    else:
        omega = 2.0 * np.pi * f
        denom = (1j * omega)**order
        denom[0] = np.inf  # avoid DC
        Y[mask] = A[mask] / denom[mask]
    return irfft(Y, n=n)

def compute_relative_motion(accel_i, accel_j, fs, f_lo, f_hi):
    """
    Compute relative displacement and velocity from acceleration measurements
    using band-limited FFT integration.  <-- Change 2
    """
    # Relative acceleration
    a_rel = (accel_i - accel_j)
    # Band-limited integrations
    v_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=1)
    x_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=2)
    return x_rel, v_rel

def compute_restoring_force(accel_i_bp, m_eff=1.0):
    """
    Restoring force proxy using band-passed acceleration at DOF i:
      R(t) ≈ -m_eff * a_i_bp
    If you have measured input force F_i, prefer R = F_i_bp - m_eff * a_i_bp.
    """
    return -m_eff * accel_i_bp

# Store results for all levels
rfs_data = {}

print("="*60)
print("MODE ISOLATION AND RFS COMPUTATION")
print("="*60)
print(f"Target mode frequency range: {freq_low:.1f} - {freq_high:.1f} Hz")
print(f"Center frequency: {target_freq:.2f} Hz")
print("="*60)
print()

for level in levels_to_compute:
    data = data_dict[level]

    # Extract acceleration signals (assumed in m/s^2)
    accel_i_raw = data[f'Acceleration{location_i}'].to_numpy()
    accel_j_raw = data[f'Acceleration{location_j}'].to_numpy()

    # Apply bandpass filter to isolate the target mode (for consistency across all signals)
    print(f"Level {level} - Applying bandpass filter ({freq_low}-{freq_high} Hz)...")
    accel_i_bp = bandpass_filter(accel_i_raw, freq_low, freq_high, fs)
    accel_j_bp = bandpass_filter(accel_j_raw, freq_low, freq_high, fs)

    # Relative motion (subtract first, then integrate band-limited)  <-- Change 2
    rel_disp, rel_vel = compute_relative_motion(accel_i_bp, accel_j_bp, fs, freq_low, freq_high)

    # Restoring force proxy (use band-passed a_i; scale by mass if known)
    restoring_force = compute_restoring_force(accel_i_bp, m_eff=m_eff)

    # Adaptive, level-dependent thresholds  <-- Change 3
    eps_v = max(vel_thresh_frac * np.max(np.abs(rel_vel)), vel_thresh_min)
    eps_x = max(disp_thresh_frac * np.max(np.abs(rel_disp)), disp_thresh_min)

    # Masks:
    # - Stiffness slice (R vs x): use near-zero velocity
    near_zero_vel_mask = np.abs(rel_vel) <= eps_v
    # - Damping slice (R vs v): use near-zero displacement
    near_zero_disp_mask = np.abs(rel_disp) <= eps_x

    # Store results
    rfs_data[level] = {
        'rel_disp': rel_disp,
        'rel_vel': rel_vel,
        'restoring_force': restoring_force,
        'near_zero_vel_mask': near_zero_vel_mask,   # used for stiffness slice
        'near_zero_disp_mask': near_zero_disp_mask   # used for damping slice
    }

    print(f"Level {level} processed:")
    print(f"  Relative displacement range: [{rel_disp.min():.6e}, {rel_disp.max():.6e}] m")
    print(f"  Relative velocity range: [{rel_vel.min():.6e}, {rel_vel.max():.6e}] m/s")
    print(f"  Restoring force proxy range: [{restoring_force.min():.6e}, {restoring_force.max():.6e}] {'N' if m_eff!=1.0 else 'm/s^2'}")
    print(f"  eps_v (|v|<=): {eps_v:.3e} m/s  |  points: {np.sum(near_zero_vel_mask)}")
    print(f"  eps_x (|x|<=): {eps_x:.3e} m    |  points: {np.sum(near_zero_disp_mask)}")
    print()

# ---------------------- Plotting ----------------------

def plot_level_pair(fig, axes, lvl_a, lvl_b, title_prefix=('a','b','c','d')):
    # Level A (stiffness slice: R vs x with |v| small)
    rel_disp_a = rfs_data[lvl_a]['rel_disp']
    rel_vel_a  = rfs_data[lvl_a]['rel_vel']
    R_a        = rfs_data[lvl_a]['restoring_force']
    mask_stiff_a = rfs_data[lvl_a]['near_zero_vel_mask']
    mask_damp_a  = rfs_data[lvl_a]['near_zero_disp_mask']

    axes[0, 0].plot(rel_disp_a[mask_stiff_a], R_a[mask_stiff_a],
                    'o', color='black', markersize=3, alpha=0.6,
                    markerfacecolor='none', markeredgewidth=0.5)
    axes[0, 0].set_xlabel('Relative Displacement [m]')
    axes[0, 0].set_ylabel('-Acceleration [m/s²]')
    axes[0, 0].set_title(f'({title_prefix[0]}) Level {lvl_a}')
    axes[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[0, 0].grid(False)

    axes[0, 1].plot(rel_vel_a[mask_damp_a], R_a[mask_damp_a],
                    'o', color='black', markersize=3, alpha=0.6,
                    markerfacecolor='none', markeredgewidth=0.5)
    axes[0, 1].set_xlabel('Relative Velocity [m/s]')
    axes[0, 1].set_ylabel('-Acceleration [m/s²]')
    axes[0, 1].set_title(f'({title_prefix[1]}) Level {lvl_a}')
    axes[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[0, 1].grid(False)

    # Level B
    rel_disp_b = rfs_data[lvl_b]['rel_disp']
    rel_vel_b  = rfs_data[lvl_b]['rel_vel']
    R_b        = rfs_data[lvl_b]['restoring_force']
    mask_stiff_b = rfs_data[lvl_b]['near_zero_vel_mask']
    mask_damp_b  = rfs_data[lvl_b]['near_zero_disp_mask']

    axes[1, 0].plot(rel_disp_b[mask_stiff_b], R_b[mask_stiff_b],
                    'o', color='black', markersize=3, alpha=0.6,
                    markerfacecolor='none', markeredgewidth=0.5)
    axes[1, 0].set_xlabel('Relative Displacement [m]')
    axes[1, 0].set_ylabel('-Acceleration [m/s²]')
    axes[1, 0].set_title(f'({title_prefix[2]}) Level {lvl_b}')
    axes[1, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[1, 0].grid(False)

    axes[1, 1].plot(rel_vel_b[mask_damp_b], R_b[mask_damp_b],
                    'o', color='black', markersize=3, alpha=0.6,
                    markerfacecolor='none', markeredgewidth=0.5)
    axes[1, 1].set_xlabel('Relative Velocity [m/s]')
    axes[1, 1].set_ylabel('-Acceleration [m/s²]')
    axes[1, 1].set_title(f'({title_prefix[3]}) Level {lvl_b}')
    axes[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[1, 1].grid(False)

# Figure 1: Levels 1 and 3
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
plot_level_pair(fig1, axes1, 1, 3, title_prefix=('a','b','c','d'))
plt.tight_layout()
plt.show()

# Figure 2: Levels 5 and 7
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
plot_level_pair(fig2, axes2, 5, 7, title_prefix=('e','f','g','h'))
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Restoring Force Surface Computation Complete")
print("="*60)
print(f"Analyzed {len(levels_to_compute)} excitation levels")
print(f"Location i (acceleration measured): {location_i}")
print(f"Location j (across connection): {location_j}")
print(f"Mode isolated: {freq_low}-{freq_high} Hz")
print(f"Velocity threshold fraction: {vel_thresh_frac:.2%} (min {vel_thresh_min:g} m/s)")
print(f"Displacement threshold fraction: {disp_thresh_frac:.2%} (min {disp_thresh_min:g} m)")
print("="*60)
# -*- coding: utf-8 -*-
"""
diffbot_trajectory_good.py

Trajectory planning for a 3-wheel differential-drive robot (front wheel rendered elsewhere).
You control the path via *wheel velocity profiles* and can directly specify the lengths of the
two straight segments:

    straight_1 (length D1) ->
    slow down (blend) ->
    90° left turn ->
    accelerate (blend) ->
    straight_2 (length D2)

Key geometric convention:
- r : wheel radius
- 2L: distance between the *centers* of the two driven side wheels (track width)

Kinematics (using wheel angular speeds ωL, ωR in rad/s):
    v   = (r/2) (ωR + ωL)
    ω   = dθ/dt = (r/(2L)) (ωR - ωL)
    ẋ   = v cosθ
    ẏ   = v sinθ

Instantaneous turning radius:
    R(t) = v(t)/ω(t)  (→ ∞ when ω→0)

Outputs:
- diffbot_trajectory.xlsx with time, ωL, ωR, v, ω, R, x, y, θ, r, L
- plots for wheel speeds, v/ω, and XY path (optional)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    'figure.figsize': (6, 4),
    'figure.dpi': 100,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 12,
    'axes.linewidth': 1.7,
    'lines.linewidth': 4,
    'grid.linestyle': ':',
    'grid.color': 'gray',
    'grid.alpha': 0.5
})
plt.close('all')

plt.close("all")

# =========================
# USER-EDITABLE PARAMS
# =========================
# Geometry (meters)
r = 0.1          # wheel radius
L = 0.5          # half track width (distance between side wheel centers is 2L)

# --- You asked to directly specify straight-segment distances:
D1_STRAIGHT = 132.5     # meters (first straight segment length)
D2_STRAIGHT = 173.5     # meters (second straight segment length)

# --- Motion profile:
V_FAST = 20          # m/s (constant speed during straight segments)
V_TURN = 4          # m/s (nominal speed during the turn)

# Time to decelerate from V_FAST -> V_TURN and accelerate back (smooth C1 ramp)
T_DECEL = 3         # seconds
T_ACCEL = 3         # seconds

# Turn specification
TURN_ANGLE = np.pi / 2     # 90 deg left (CCW)
T_TURN_TOTAL = 2.50        # total duration of the turn phase (includes ω ramps)

# Within the turn phase, ω is: ramp up -> plateau -> ramp down
TURN_RAMP_FRAC = 0.25      # fraction of turn phase used for each ω ramp (0..0.5)
# Example: 0.25 => 25% ramp-up, 50% plateau, 25% ramp-down

# Discretization
DT = 0.01

# Output
OUT_PATH = Path(__file__).with_name("diffbot_trajectory.xlsx")
SHEET_NAME = "data"

# Plots
PLOT_RESULTS = True
# =========================


def smoothstep(u: np.ndarray) -> np.ndarray:
    """C1 smooth step: 0->1 with zero slope at endpoints."""
    u = np.clip(u, 0.0, 1.0)
    return 3*u**2 - 2*u**3


def ramp_profile(t: np.ndarray, t0: float, t1: float, y0: float, y1: float) -> np.ndarray:
    """C1 ramp from y0 to y1 over [t0, t1]."""
    y = np.full_like(t, y0, dtype=float)
    if t1 <= t0:
        y[t >= t0] = y1
        return y
    mask = (t >= t0) & (t <= t1)
    u = (t[mask] - t0) / (t1 - t0)
    s = smoothstep(u)
    y[mask] = (1 - s) * y0 + s * y1
    y[t > t1] = y1
    return y


def omega_turn_profile(t: np.ndarray, t_start: float, t_end: float, angle: float, ramp_frac: float) -> np.ndarray:
    """
    Build ω(t) over [t_start, t_end]:
        0 outside
        ramp up -> plateau -> ramp down inside
    Scale so that ∫ ω dt = angle exactly.
    """
    w = np.zeros_like(t, dtype=float)
    T = max(1e-9, t_end - t_start)

    ramp_frac = float(np.clip(ramp_frac, 0.0, 0.5))
    Tr = ramp_frac * T               # ramp duration (each)
    Tp = T - 2.0 * Tr                # plateau duration

    # Shape function g(t) in [0,1], then scale by ω_max
    g = np.zeros_like(t, dtype=float)
    # ramp up
    if Tr > 1e-9:
        mask = (t >= t_start) & (t <= t_start + Tr)
        u = (t[mask] - t_start) / Tr
        g[mask] = smoothstep(u)
    # plateau
    mask = (t >= t_start + Tr) & (t <= t_start + Tr + Tp)
    g[mask] = 1.0
    # ramp down
    if Tr > 1e-9:
        mask = (t >= t_start + Tr + Tp) & (t <= t_end)
        u = (t[mask] - (t_start + Tr + Tp)) / Tr
        g[mask] = 1.0 - smoothstep(u)

    # Scale so that integral equals requested angle
    # ∫ ω dt = ω_max ∫ g dt  => ω_max = angle / ∫ g dt
    area = np.trapz(g, t)
    if abs(area) < 1e-12:
        raise ValueError("Turn profile area is ~0; increase T_TURN_TOTAL or adjust TURN_RAMP_FRAC.")
    w_max = angle / area
    w = w_max * g
    return w


def simulate_diff_drive(t: np.ndarray, wl: np.ndarray, wr: np.ndarray, r: float, L: float,
                        x0=0.0, y0=0.0, th0=0.0):
    """Forward Euler integration of differential-drive kinematics."""
    x = np.zeros_like(t, dtype=float)
    y = np.zeros_like(t, dtype=float)
    th = np.zeros_like(t, dtype=float)

    x[0], y[0], th[0] = x0, y0, th0

    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        v = 0.5 * r * (wr[i] + wl[i])
        w = (r / (2.0 * L)) * (wr[i] - wl[i])

        x[i+1] = x[i] + v * np.cos(th[i]) * dt
        y[i+1] = y[i] + v * np.sin(th[i]) * dt
        th[i+1] = th[i] + w * dt

    return x, y, th


def main():
    # -------------------------
    # 1) Compute segment times from the distances you specified
    # -------------------------
    # Straight 1 at V_FAST
    T1 = D1_STRAIGHT / max(1e-9, V_FAST)

    # Decel ramp duration (we explicitly allocate time; distance will be whatever results)
    Tdec = max(0.0, T_DECEL)

    # Turn duration
    Tturn = max(1e-6, T_TURN_TOTAL)

    # Accel ramp duration
    Tacc = max(0.0, T_ACCEL)

    # Straight 2 at V_FAST
    T2 = D2_STRAIGHT / max(1e-9, V_FAST)

    # Time boundaries
    t0 = 0.0
    t1 = t0 + T1
    t2 = t1 + Tdec
    t3 = t2 + Tturn
    t4 = t3 + Tacc
    t5 = t4 + T2

    t = np.arange(t0, t5 + DT, DT)

    # -------------------------
    # 2) Build v(t): fast const -> decel -> turn const -> accel -> fast const
    # -------------------------
    v = np.full_like(t, V_FAST, dtype=float)

    # Decel ramp (t1->t2): V_FAST -> V_TURN
    v = np.minimum(v, ramp_profile(t, t1, t2, V_FAST, V_TURN))  # replaces values after t1
    # Turn constant (t2->t3): hold V_TURN
    v[(t >= t2) & (t <= t3)] = V_TURN
    # Accel ramp (t3->t4): V_TURN -> V_FAST
    v_after = ramp_profile(t, t3, t4, V_TURN, V_FAST)
    v[t >= t3] = v_after[t >= t3]
    # Straight 2 already at V_FAST (t4->t5)

    # -------------------------
    # 3) Build ω(t): only nonzero during turn phase (t2->t3)
    # -------------------------
    omega = omega_turn_profile(t, t_start=t2, t_end=t3, angle=TURN_ANGLE, ramp_frac=TURN_RAMP_FRAC)

    # -------------------------
    # 4) Convert (v, ω) to wheel angular speeds (ωL, ωR)
    #    v = r/2 (ωR + ωL)
    #    ω = r/(2L) (ωR - ωL)
    # => ωR = (v + ωL_body) / r, with ωL_body = ω*L
    #    ωL = (v - ω*L) / r
    # -------------------------
    wr = (v + omega * L) / r
    wl = (v - omega * L) / r
    wl_acc = np.gradient(wl, t)  # left wheel acceleration
    wr_acc = np.gradient(wr, t) 
    

    # -------------------------
    # 5) Simulate pose
    # -------------------------
    x, y, th = simulate_diff_drive(t, wl, wr, r=r, L=L)

    # -------------------------
    # 6) Turning radius
    # -------------------------
    R = np.full_like(t, np.inf, dtype=float)
    mask_turn = np.abs(omega) > 1e-8
    R[mask_turn] = v[mask_turn] / omega[mask_turn]

    # -------------------------
    # 7) Save
    # -------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    df = pd.DataFrame({
        "time": t,
        "wl": wl,
        "wr": wr,
        "v": v,
        "omega": omega,
        "R": R,
        "x": x,
        "y": y,
        "theta": th,
        "r": np.full_like(t, r, dtype=float),
        "L": np.full_like(t, L, dtype=float),
        "dt": np.full_like(t, DT, dtype=float),
        "D1": np.full_like(t, D1_STRAIGHT, dtype=float),
        "D2": np.full_like(t, D2_STRAIGHT, dtype=float),
    })
    df.to_excel(OUT_PATH, index=False, sheet_name=SHEET_NAME)
    print(f"Wrote: {OUT_PATH}")

    # -------------------------
    # 8) Plots
    # -------------------------
    LABEL_SIZE = 18
    TICK_SIZE  = 16
    LEGEND_SIZE = 16
    
    if PLOT_RESULTS:
        # First plot for wheel angular velocities and accelerations
        fig1, axs = plt.subplots(2, 1, figsize=(10, 10))  # Adjusted figure size for subplots
        axs[0].plot(t, wl, label=r"left wheel $\omega_L$")
        axs[0].plot(t, wr, label=r"right wheel $\omega_R$")
        axs[0].set_ylabel("wheel angular velocity (rad/s)",
                          fontsize=LABEL_SIZE,
                          fontweight="bold")
        axs[0].legend(prop={"size": LEGEND_SIZE, "weight": "bold"})
        axs[0].tick_params(axis="both", labelsize=TICK_SIZE)
        for label in axs[0].get_xticklabels() + axs[0].get_yticklabels():
            label.set_fontweight("bold")
        axs[0].grid(True)

        axs[1].plot(t, wl_acc, label=r"left wheel $\dot{\omega_L}$")
        axs[1].plot(t, wr_acc, label=r"right wheel $\dot{\omega_R}$")
        axs[1].set_ylabel("wheel angular acceleration (rad/s²)",
                          fontsize=LABEL_SIZE,
                          fontweight="bold")
        axs[1].set_xlabel("time (s)", fontsize=LABEL_SIZE, fontweight="bold")
        axs[1].legend(prop={"size": LEGEND_SIZE, "weight": "bold"})
        axs[1].tick_params(axis="both", labelsize=TICK_SIZE)
        for label in axs[1].get_xticklabels() + axs[1].get_yticklabels():
            label.set_fontweight("bold")
        axs[1].grid(True)

        fig1.tight_layout()
        plt.show()
        
        # Second plot for mobile robot body rates
        fig2 = plt.figure(figsize=(10, 10))  # Adjusted figure size for the second plot
        ax2 = fig2.add_subplot(111)
        ax2.plot(t, v, label="v (m/s)")
        ax2.plot(t, omega, label=r"turning angular velocity $\omega$ (rad/s)")
        ax2.set_ylabel("mobile robot body rate",
                        fontsize=LABEL_SIZE,
                        fontweight="bold")
        ax2.set_xlabel("time (s)",
                        fontsize=LABEL_SIZE,
                        fontweight="bold")
        ax2.tick_params(axis="both", labelsize=TICK_SIZE)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight("bold")
        ax2.grid(True)
        ax2.legend(prop={"size": LEGEND_SIZE, "weight": "bold"})
        
        fig2.tight_layout()
        plt.show()


        # plt.figure(figsize=(8, 8))
        # plt.plot(x, y)
        # plt.axis("equal")
        # plt.xlabel("x (m)")
        # plt.ylabel("y (m)")
        # plt.title("Planned path (from wheel velocity profile)")
        # plt.grid(True)
        # plt.show()


if __name__ == "__main__":
    main()

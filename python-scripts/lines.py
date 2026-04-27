import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.exposure import rescale_intensity

# ------------------------------------------------------------
# clc, clear, close all
# ------------------------------------------------------------
plt.close("all")

# ------------------------------------------------------------
# img = imread(...)
# ------------------------------------------------------------
img_bgr = cv2.imread(r"C:\Users\littl\Desktop\p3_pics\map.png")
if img_bgr is None:
    raise RuntimeError("Image not found. Check your path.")

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

Nr, Nc = img.shape[:2]

# ------------------------------------------------------------
# dor_red = 1/2;
# ------------------------------------------------------------
dor_red = 1 / 2

# ------------------------------------------------------------
# red mask (vectorized version of your loops)
# ------------------------------------------------------------
r = img[:, :, 0].astype(np.float32)
g = img[:, :, 1].astype(np.float32)
b = img[:, :, 2].astype(np.float32)

red = np.zeros((Nr, Nc), dtype=np.float32)

mask = (
    (r > (1 - dor_red) * 255) &
    (g < dor_red * 255) &
    (b < dor_red * 255)
)
red[mask] = 255.0

fig = plt.figure(frameon=False, figsize=(18.2, 18.2), dpi=200)
fig.patch.set_visible(False)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(red, cmap="gray")
ax.axis("off")
plt.show()


#%%

# ------------------------------------------------------------
# conv2 edges
# ------------------------------------------------------------
K = np.array([[0.5, 0.0, -0.5]], dtype=np.float32)

v_edge = cv2.filter2D(red, -1, K, borderType=cv2.BORDER_CONSTANT)
h_edge = cv2.filter2D(red, -1, K.T, borderType=cv2.BORDER_CONSTANT)

edge = np.sqrt(h_edge**2 + v_edge**2)

# ------------------------------------------------------------
# threshold = 0; bi_edge = edge > threshold;
# ------------------------------------------------------------
threshold = 0
bi_edge = edge > threshold

# ------------------------------------------------------------
# Plot: bi_edge (NO axis, NO padding)
# ------------------------------------------------------------
fig = plt.figure(frameon=False, figsize=(18.2, 18.2), dpi=200)
fig.patch.set_visible(False)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(bi_edge, cmap="gray")
ax.axis("off")
plt.show()

#%%

# ------------------------------------------------------------
# Hough transform
# ------------------------------------------------------------
theta_deg = np.arange(-90.0, 89.0 + 1e-12, 0.25)
theta = np.deg2rad(theta_deg)

rho_resolution = 0.25
scale = int(round(1.0 / rho_resolution))

if scale > 1:
    bi_u8 = (bi_edge.astype(np.uint8) * 255)
    bi_up = cv2.resize(bi_u8, (Nc * scale, Nr * scale), interpolation=cv2.INTER_NEAREST)
    bi_for_hough = bi_up.astype(bool)
else:
    bi_for_hough = bi_edge

H, T, R = hough_line(bi_for_hough, theta=theta)
accum, Tpeaks, Rpeaks = hough_line_peaks(
    H, T, R,
    num_peaks=20,          # allow extras; suppression will collapse
    min_distance=16,        # rho-bin suppression radius (increase to collapse more)
    min_angle=16,           # theta-bin suppression radius (increase to collapse more)
    threshold=0.5*np.max(H)  # avoid noise peaks
)

# ------------------------------------------------------------
# Hough accumulator plot with contrast stretch + axes
# ------------------------------------------------------------

# MATLAB-like contrast enhancement (imadjust(rescale(H)))
lo, hi = np.percentile(H, (2, 98))
Hshow = rescale_intensity(H, in_range=(lo, hi), out_range=(0, 1))

# make rho axis consistent with the original image scale
R_plot = R / scale if scale > 1 else R

# ALSO scale the peak rhos to match the plotted rho axis
Rpeaks_plot = Rpeaks / scale if scale > 1 else Rpeaks

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)

ax.imshow(
    Hshow,
    cmap="gray",
    aspect="auto",
    extent=[np.rad2deg(T[0]), np.rad2deg(T[-1]), R_plot[-1], R_plot[0]]
)

# mark peaks
ax.plot(
    np.rad2deg(Tpeaks),
    Rpeaks_plot,
    "rs",
    markersize=8,
    markerfacecolor="none",
    markeredgewidth=1.5
)

ax.set_xlabel(r'$\theta$ (deg)', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$\rho$ (pixels)', fontsize=16, fontweight='bold')

ax.tick_params(axis='both', which='major', labelsize=14)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')
    
plt.tight_layout()
plt.show()


#%%
# ------------------------------------------------------------
# Print: equations of all lines (from peaks)
# Standard normal form in image coords:
#   x*cos(theta) + y*sin(theta) = rho
# And slope form (if sin(theta) != 0):
#   y = (rho - x*cos(theta)) / sin(theta)
# ------------------------------------------------------------
print("\n=== Line equations from Hough peaks ===")
for i, (ang, rho) in enumerate(zip(Tpeaks, Rpeaks), start=1):
    th_deg = np.rad2deg(ang)
    c = np.cos(ang)
    s = np.sin(ang)

    rho0 = rho / scale if scale > 1 else rho  # rho in original pixels

    print(f"\nLine {i}:")
    print(f"  theta = {th_deg:.6f} deg, rho = {rho0:.6f}")
    print(f"  Normal form:  x*{c:.12f} + y*{s:.12f} = {rho0:.12f}")

    if abs(s) > 1e-12:
        m = -(c / s)
        b_int = rho0 / s
        print(f"  Slope form:   y = ({rho0:.12f} - x*{c:.12f}) / {s:.12f}")
        print(f"               y = {m:.12f}*x + {b_int:.12f}")
    else:
        if abs(c) > 1e-12:
            x0 = rho0 / c
            print(f"  Vertical form (sin≈0): x = {x0:.12f}")
        else:
            print("  Degenerate (sin≈0 and cos≈0) — should not happen for real peaks.")

#%%
import matplotlib.cm as cm
import matplotlib.colors as colors
# ------------------------------------------------------------
# Deterministic finite segment extraction using Hough peaks
# (MATLAB houghlines-like idea)
# ------------------------------------------------------------

dist_tol = 1.5     # how close (in pixels) an edge pixel must be to the line
fill_gap = 20      # like MATLAB FillGap
min_length = 100   # like MATLAB MinLength

ys, xs = np.nonzero(bi_edge)  # edge pixel coordinates (row=y, col=x)
xs = xs.astype(np.float64)
ys = ys.astype(np.float64)

segments = []  # list of ((x0,y0),(x1,y1), rho, theta)

for theta, rho in zip(Tpeaks, Rpeaks):
    c = np.cos(theta)
    s = np.sin(theta)
    rho0 = rho / scale if scale > 1 else rho

    # Signed perpendicular distance to line: x*c + y*s - rho
    # (since sqrt(c^2+s^2)=1)
    d = xs * c + ys * s - rho0
    sel = np.abs(d) <= dist_tol
    if not np.any(sel):
        continue

    xk = xs[sel]
    yk = ys[sel]

    # Unit direction vector along the line is perpendicular to (c,s):
    # v = (-s, c)
    t = xk * (-s) + yk * (c)  # 1D coordinate along the line

    order = np.argsort(t)
    t = t[order]
    xk = xk[order]
    yk = yk[order]

    # Split into runs by gap along t
    # (gap in pixels along the line; approximates FillGap)
    gaps = np.diff(t)
    breaks = np.where(gaps > fill_gap)[0]

    start = 0
    for b in np.r_[breaks, len(t) - 1]:
        end = b + 1  # inclusive end index in Python slice is end
        if end - start < 2:
            start = end
            continue

        # Endpoints of this run (in image coords)
        x0, y0 = xk[start], yk[start]
        x1, y1 = xk[end - 1], yk[end - 1]

        # Segment length (Euclidean)
        seg_len = np.hypot(x1 - x0, y1 - y0)
        if seg_len >= min_length:
            segments.append(((x0, y0), (x1, y1), rho0, theta))

        start = end

# ------------------------------------------------------------
# Plot extracted finite segments on bi_edge (axis off, no padding)
# ------------------------------------------------------------
fig = plt.figure(frameon=False, figsize=(18.2, 18.2), dpi=200)
fig.patch.set_visible(False)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_axes([0, 0, 1, 1])

ax.imshow(bi_edge, cmap="gray")


colors = ["r", "yellow", "yellow", "r", "yellow","r", "yellow", "r"]

for i, ((x0, y0), (x1, y1), rho, theta) in enumerate(segments):
    ax.plot([x0, x1], [y0, y1],
            linewidth=2,
            color=colors[i % len(colors)])

ax.axis("off")
plt.show()

print(f"Segments found: {len(segments)}")

print("\nLine equations (normal form) for each segment:")
for i, ((x0, y0), (x1, y1), rho, theta) in enumerate(segments, start=1):
    c = np.cos(theta)
    s = np.sin(theta)
    print(f"{i:02d}: x*{c:.6f} + y*{s:.6f} = {rho:.6f}  |  segment endpoints: ({x0:.1f},{y0:.1f}) -> ({x1:.1f},{y1:.1f})")


print(f"\nNum extracted segments: {len(segments)}")


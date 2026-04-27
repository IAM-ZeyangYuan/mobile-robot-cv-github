import numpy as np
import matplotlib.pyplot as plt

from imageio.v3 import imread
from skimage.color import rgb2gray
from skimage.morphology import erosion, dilation
from scipy.ndimage import binary_fill_holes



# def markers_detection():
# --- MATLAB: clc, clear, close all ---
# (In Python, we just start fresh; figures are controlled by matplotlib.)

# --- MATLAB: img = imread('CWMap.jpg'); ---
img = imread(r"C:\Users\littl\Desktop\p3_pics\map.png")  # expects RGB image
if img.ndim == 3 and img.shape[2] == 4:
    img = img[:, :, :3]
    
r = img[:, :, 0]
g = img[:, :, 1]
b = img[:, :, 2]

saize = img.shape
Nr = saize[0]
Nc = saize[1]


dor_blue = 1 / 2            
dor_green = 1 / 1.5            
            
blue = ((b > (1 - dor_blue) * 255) &
        (g < dor_blue * 255) &
        (r < dor_blue * 255)).astype(np.uint8)

green = ((g > (1 - dor_green) * 255) &
         (b < dor_green * 255) &
         (r < dor_green * 255)).astype(np.uint8)            

fig = plt.figure(1, frameon=False, figsize=(18.2, 18.2), dpi=200)
fig.patch.set_visible(False)

ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(blue, cmap="gray")
ax.axis("off")

fig = plt.figure(2, frameon=False, figsize=(18.2, 18.2), dpi=200)
fig.patch.set_visible(False)

ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(green, cmap="gray")
ax.axis("off")

plt.show()



#%%
# --- MATLAB: R = 20; K = zeros(2*R+1,2*R+1); meshgrid; circle fill ---
#kernel
R = 20
K = np.zeros((2 * R + 1, 2 * R + 1), dtype=np.uint8)
x, y = np.meshgrid(np.arange(1, 2 * R + 2), np.arange(1, 2 * R + 2))  # 1..2R+1 inclusive
in_circle = np.where(np.round((x - (R+1)) ** 2 + (y - (R+1)) ** 2 - R**2) <= 0)
K[in_circle] = 1

# --- MATLAB: ero_blue = imerode(blue,K); dil_blue = imdilate(ero_blue,K); ---
# skimage expects boolean-like; we keep as 0/1 uint8 and convert to bool when applying.

fig = plt.figure(3, frameon=False)
fig.patch.set_visible(False)

ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(K, cmap="gray", interpolation="nearest")
ax.axis("off")

plt.show()



#%%


ero_blue = erosion(blue.astype(bool), K.astype(bool)).astype(np.uint8)
dil_blue = dilation(ero_blue.astype(bool), K.astype(bool)).astype(np.uint8)

ero_green = erosion(green.astype(bool), K.astype(bool)).astype(np.uint8)
dil_green = dilation(ero_green.astype(bool), K.astype(bool)).astype(np.uint8)

#%%

# Erosion (blue)
fig = plt.figure(4, frameon=False, figsize=(18.2, 18.2), dpi=200)
fig.patch.set_visible(False)

ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(ero_blue, cmap="gray")
ax.axis("off")

plt.show()


# Dilation (blue)
fig = plt.figure(5, frameon=False, figsize=(18.2, 18.2), dpi=200)
fig.patch.set_visible(False)

ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(dil_blue, cmap="gray")
ax.axis("off")

plt.show()


# Erosion (green)
fig = plt.figure(6, frameon=False, figsize=(18.2, 18.2), dpi=200)
fig.patch.set_visible(False)

ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(ero_green, cmap="gray")
ax.axis("off")

plt.show()


# Dilation (green)
fig = plt.figure(7, frameon=False, figsize=(18.2, 18.2), dpi=200)
fig.patch.set_visible(False)

ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(dil_green, cmap="gray")
ax.axis("off")

plt.show()




#%%
# --- MATLAB perimeter extraction on dil_blue ---
# MATLAB indexes r-1,r+1,c-1,c+1 even on borders.
# To preserve the exact neighbor logic without changing loop bounds,
# we pad with a 1-pixel zero border and run the same checks.
dil_blue_p = np.pad(dil_blue, ((1, 1), (1, 1)), mode="constant", constant_values=0)

Per = np.zeros((Nr, Nc), dtype=np.uint8)
for r in range(Nr):
    for c in range(Nc):
        rp = r + 1
        cp = c + 1
        if (dil_blue_p[rp, cp] == 1) and not (
            (dil_blue_p[rp - 1, cp] == 1)
            and (dil_blue_p[rp + 1, cp] == 1)
            and (dil_blue_p[rp, cp - 1] == 1)
            and (dil_blue_p[rp, cp + 1] == 1)
        ):
            Per[r, c] = 1

# --- MATLAB: perimeter walking / labeling ---
Ns = 0
rnblist = [-1, -1, -1, 0, 0, 1, 1, 1]
cnblist = [-1, 0, 1, -1, 1, -1, 0, 1]

Taken = np.zeros((Nr, Nc), dtype=np.uint8)

# Use padding here too so neighbor checks during tracing can't go out of bounds.
Per_p = np.pad(Per, ((1, 1), (1, 1)), mode="constant", constant_values=0)
Taken_p = np.pad(Taken, ((1, 1), (1, 1)), mode="constant", constant_values=0)

Labels = []     # list of shapes; each shape is a list of (row, col) in MATLAB-style 1-based coordinates
Npixels = []    # number of pixels per shape (per MATLAB variable)

for r in range(Nr):
    for c in range(Nc):
        rp = r + 1
        cp = c + 1
        if (Per_p[rp, cp] == 1) and (Taken_p[rp, cp] == 0):
            Ns += 1
            npix = 1

            # store MATLAB-style coordinates (1-based)
            shape_coords = [(r + 1, c + 1)]
            Taken_p[rp, cp] = 1

            rn = r + 1 + 1  # MATLAB: rn = r+1 (with 1-based r), then cn=c+1; replicated in 1-based terms
            cn = c + 1 + 1
            ro = r + 1
            co = c + 1

            # Walk until we get back to seed (r+1, c+1) in 1-based coordinates.
            while (rn != (r + 1)) or (cn != (c + 1)):
                found = False
                for i in range(8):
                    rn = ro + rnblist[i]
                    cn = co + cnblist[i]
                    # 8 directions
                    
                    # translate to padded array indices (since Per_p is padded around 1..Nr,1..Nc)
                    rnp = rn + 1 - 1  # rn is 1-based in our variables; Per_p uses same 1..Nr mapped to 1..Nr, plus pad
                    cnp = cn + 1 - 1

                    if (Per_p[rnp, cnp] == 1) and (Taken_p[rnp, cnp] == 0):
                        npix += 1
                        shape_coords.append((rn, cn))
                        Taken_p[rnp, cnp] = 1
                        ro = rn
                        co = cn
                        #as long as it finds one of the 8 directions is also a perimetral pixel, it walks, fuck the rest of the directions

                        # MATLAB: if npix==3, Taken(r,c)=0; end
                        # Here r,c seed is at (r+1,c+1) in 1-based; in padded it's (r+1,c+1).
                        if npix == 3:
                            Taken_p[rp, cp] = 0
                        # after the second step it release the seed from the taken_p so that it can be walked (that if statement right above) and finish the whoel perimeter walk
                        
                        found = True
                        break
                if not found:
                    break 

            # MATLAB: npix = npix - 1; (walked back into seed)
            npix = npix - 1
            Npixels.append(npix)
            Labels.append(shape_coords)

print(Ns)

#%% --- MATLAB: figure(3) grayscale + im2bw threshold 0.5; show; hold on ---
plt.figure(3)
grayImage = rgb2gray(img)  # returns float in [0,1]
bb = (grayImage > 0.5).astype(np.uint8)
plt.imshow(bb, cmap="gray")
plt.title("Figure 3")
plt.axis("off")



# --- MATLAB: blue centroids computed from perimeter coordinate lists ---
centroids_blue = np.zeros((Ns, 2), dtype=float)

for i in range(Ns):
    coords = Labels[i][:Npixels[i]]  # first Npixels
    rows = np.array([rc[0] for rc in coords], dtype=float)  # 1-based
    cols = np.array([rc[1] for rc in coords], dtype=float)  # 1-based

    # MATLAB: m00 = nnz(rows); since rows are 1..Nr, nnz == len(rows)
    m00 = np.count_nonzero(rows)
    m10 = np.sum(cols)
    m01 = np.sum(rows)

    uc = m10 / m00
    vc = m01 / m00

    centroids_blue[i, 0] = uc
    centroids_blue[i, 1] = vc

    plt.scatter(uc, vc, linewidths=4)  # MATLAB used 'b' color explicitly; matplotlib defaults

# --- MATLAB: green centroid computed over all dilated-green pixels ---
m_g00 = 0
m_g10 = 0
m_g01 = 0
for r in range(Nr):
    for c in range(Nc):
        if dil_green[r, c] == 1:
            m_g00 += 1
            m_g10 += (c + 1)  # MATLAB column index is 1-based
            m_g01 += (r + 1)  # MATLAB row index is 1-based

centroids_green = np.array([m_g10 / m_g00, m_g01 / m_g00], dtype=float)
plt.scatter(centroids_green[0], centroids_green[1])  # MATLAB used 'g' color explicitly; matplotlib defaults

plt.show()

#return centroids_blue, centroids_green, Ns


# if __name__ == "__main__":
#     markers_detection()



#%%
from collections import deque


def fill_from_perimeter(perim_mask):
    """
    perim_mask : binary image, 1 = perimeter, 0 = background
    returns a filled binary region
    """

    h, w = perim_mask.shape

    visited = np.zeros_like(perim_mask, dtype=bool)

    q = deque()

    # start flood fill from the image borders (background)
    for c in range(w):
        if perim_mask[0, c] == 0:
            q.append((0, c))
            visited[0, c] = True
        if perim_mask[h-1, c] == 0:
            q.append((h-1, c))
            visited[h-1, c] = True

    for r in range(h):
        if perim_mask[r, 0] == 0:
            q.append((r, 0))
            visited[r, 0] = True
        if perim_mask[r, w-1] == 0:
            q.append((r, w-1))
            visited[r, w-1] = True

    # 4-connected flood fill
    while q:
        r, c = q.popleft()

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr = r + dr
            cc = c + dc

            if 0 <= rr < h and 0 <= cc < w:
                if (not visited[rr, cc]) and perim_mask[rr, cc] == 0:
                    visited[rr, cc] = True
                    q.append((rr, cc))

    # visited == reachable background
    # everything else that is not perimeter is interior
    filled = np.zeros_like(perim_mask, dtype=np.uint8)
    filled[(~visited) | (perim_mask == 1)] = 1

    return filled


# ---------------------------------------------------------
# Plot each separated blue marker, filled
# ---------------------------------------------------------

for k in range(Ns):

    coords = Labels[k][:Npixels[k]]

    rows = np.array([p[0] for p in coords]) - 1
    cols = np.array([p[1] for p in coords]) - 1

    perim = np.zeros((Nr, Nc), dtype=np.uint8)
    perim[rows, cols] = 1

    filled = fill_from_perimeter(perim)



    # plot without padding, axis off
    fig = plt.figure(frameon=False, figsize=(18.2, 18.2), dpi=200)
    fig.patch.set_visible(False)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(filled, cmap="gray", interpolation="nearest")

    ax.axis("off")

    plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

radii = np.zeros(Ns)

for k in range(Ns):

    coords = Labels[k][:Npixels[k]]

    # MATLAB-style coordinates (row, col)
    rows = np.array([p[0] for p in coords], dtype=float)
    cols = np.array([p[1] for p in coords], dtype=float)

    uc = centroids_blue[k, 0]   # column centroid
    vc = centroids_blue[k, 1]   # row centroid

    # radius = mean distance to centroid
    d = np.sqrt((cols - uc)**2 + (rows - vc)**2)
    radii[k] = d.mean()

    # plot
    fig, ax = plt.subplots()

    ax.plot(cols, rows, '.', markersize=2)
    ax.scatter(uc, vc, s=10)

    ax.invert_yaxis()
    ax.axis('off')

    # remove all padding / white borders
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.margins(0)

    plt.show()

    print(f"Blue marker {k+1} radius (pixels): {radii[k]}")


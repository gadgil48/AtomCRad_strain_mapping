# Simple implementation to generate some reference diffraction disk image
# Based on code by Daniel Mason 28/08/2020
# Adapted by Sanket Gadgil 30/10/2020

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

# Image dimensions (in pixels)
Nx, Ny = 128, 128

# Reciprocal lattice (in pixels)
a = np.array([[22.0, 0.0], [0.0, 22.0]])

# Image rotation angle (in radians) and matrix
theta = 0.0
theta = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

# Diffraction disk (in pixels)
r = 8.0

# Maximum and minimum diffraction disk amplitudes
disk_max, disk_min = 5.25, 3.5

# Maximum and minimum non-disk background amplitudes
max_back, min_back = 4.35, 3.1

# Gaussian standard division for diffraction disk amplitude
sigma = 20.0
# Gaussian standard division for non-disk background
sigma2 = 32.0

# X-Y offset (in pixels)
delta = np.array([0.0, 0.0])

img = np.zeros((Nx, Ny))

# Inverted reciprocal lattice and rotation matrices
inv_a = np.linalg.inv(a)
inv_theta = np.linalg.inv(theta)

# Main loop over all pixels
for ix in range(-Nx//2, Nx//2):
    for iy in range(-Ny//2, Ny//2):
        # Position of ix and iy with offset removed
        xx = np.array((ix-delta[0], iy-delta[1]), dtype=np.int16)

        # Apply rotation matrix to ix-iy
        xx = np.dot(theta, xx)
        xx = np.rint(xx)  # rounding to nearest integer since using pixels

        # Position of ix-iy in reciprocal space
        x0 = inv_a[0, :]*xx[0] + inv_a[1, :]*xx[1]

        # Find nearest disk centre
        x0 = np.rint(x0)
        x0 = a[0, :]*x0[0] + a[1, :]*x0[1]

        # Find offset of pixel to disk centre
        dx = xx - x0

        # Find distance (squared) of the pixel to the disk centre
        rr = dx[0]*dx[0] + dx[1]*dx[1]

        # If pixel is outside disk then apply the amplitude associated with the
        # background Gaussian (with sigma2)
        if (rr > r*r):
            img[ix+Nx//2, iy+Ny//2] = min_back
            ff = np.exp(-np.linalg.norm(xx)**2/(2*sigma2*sigma2))
            ff *= (max_back - img[ix+Nx//2, iy+Ny//2])
            img[ix+Nx//2, iy+Ny//2] += ff
            continue

        # Find distance to the centre of the image
        rr = xx[0]*xx[0] + xx[1]*xx[1]

        # Calculate amplitude in accordance with a slow gaussian decay
        ff = np.exp(- rr/(2*sigma*sigma))

        # Normalise between max/min of the disk amplitudes
        img[ix+Nx//2, iy+Ny//2] = disk_min
        ff *= (disk_max - img[ix+Nx//2, iy+Ny//2])
        img[ix+Nx//2, iy+Ny//2] += ff

# Save data (change filepath as necessary)
filepath = "/mnt/c/Users/Sanket_Work/Documents/Project 1 Image Analysis/\
Sample data/ref_patt_synth.csv"
np.savetxt(filepath, img, fmt='%f', delimiter=',', newline='\n')

# Plot data
plt.imshow(img, cmap='jet')
plt.colorbar()
plt.show()

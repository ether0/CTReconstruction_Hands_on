import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift

# 1. Design a Ram-lak filter
# see Reconstruction.Filter
num_points = 129
dx = 0.1
w0 = 1/(2*dx) # Nyquist frequency

x_coord = np.linspace(-(num_points-1) * dx/2, (num_points-1) * dx/2, num_points)
x = np.arange(0, num_points) - (num_points - 1) / 2.0
w_coord = x / (num_points * dx)

h = np.zeros(len(x))
#################### Your code here ####################

########################################################

filter_RL = np.abs(fftshift(fft(h)))
filter_RL[np.where(abs(w_coord) > w0)] = 0

# 2. Design a ideal ramp filter in the frequency domain
# extend w_coord to > w0
padding_n = 10000
dw = w_coord[1] - w_coord[0]
w_coord_ext_l = np.linspace(w_coord[0] - dw * padding_n, w_coord[0], padding_n)
w_coord_ext_r = np.linspace(w_coord[-1], w_coord[-1] + dw * padding_n, padding_n)
w_coord_ext = np.concatenate((w_coord_ext_l, w_coord, w_coord_ext_r))

filter_ramp = np.zeros_like(w_coord_ext)
#################### Your code here ####################

########################################################
filter_ramp /= dx 
filter_ramp[np.where(abs(w_coord_ext) > w0)] = 0

h_ramp = np.real(ifft(ifftshift(filter_ramp)))
h_ramp = h_ramp / np.max(h_ramp) * np.max(h) # Normalize
x_coord_ext = np.fft.fftfreq(len(filter_ramp), d=(w_coord_ext[1] - w_coord_ext[0]))

# 3. Plot
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(x_coord_ext, h_ramp, label='Ramp', linestyle = 'none', marker='o', markersize = 3)
ax[0].plot(x_coord, h, label='Ram-Lak', linestyle = 'none', marker='o', markersize = 3)
ax[0].set_title('Spatial Domain')
ax[0].set_xlabel('Position (x)')
# ax[0].set_xlim(-2.5, 2.5)
ax[0].legend()

ax[1].plot(w_coord_ext, filter_ramp, label='Ramp', marker='o', markersize = 3)
ax[1].plot(w_coord, filter_RL, label='Ram-Lak', linestyle = 'none', marker='o', markersize = 3)
ax[1].set_title('Frequency Domain')
ax[1].set_xlabel('Frequency (w)')
# ax[1].set_xlim(-10, 10)
ax[1].legend()

plt.tight_layout()
plt.show()
from Reconstruction_pycuda import Reconstruction
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift

# 1. Parameters
Source = np.array([0, 1000.0, 0])
Detector = np.array([0, -500.0, 0])
Origin = np.array([0, 0, 0])
nu = 512
du = 0.5

ZeroPaddedLength = int(2 ** (np.ceil(np.log2(2.0 * (nu - 1)))))
R = np.sqrt(np.sum((Source - Origin) ** 2.0))
D = np.sqrt(np.sum((Source - Detector) ** 2.0)) - R
N = ZeroPaddedLength + 1
pixel_size = du * R / (D + R)
cutoff = 0.8

# 2. Design filters
filter_dict = {}
filter_list = ['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann']

# Ideal ramp filter
x = np.arange(0, N) - (N - 1) / 2.0
w = x[0:-1] / ((N - 1) * pixel_size)

filter = 1/2 * np.abs(w)
filter /= pixel_size

filter[np.where(abs(w) > cutoff / (2.0 * pixel_size))] = 0
filter_dict['ideal ramp'] = filter

for FilterType in filter_list:
    filter = Reconstruction.Filter(N, pixel_size, FilterType, cutoff)
    filter_dict[FilterType] = filter

# 3. Plot
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# all filters
for key, value in filter_dict.items():
    ax[0].plot(w, value, label=key)
ax[0].set_title('All filters')
ax[0].legend()

# ramp and ram-lak, zoomed in
ax[1].axhline(y=0, color='k', linewidth=0.5)
ax[1].axvline(x=0, color='k', linewidth=0.5)

for key, value in filter_dict.items():
    if key in ['ideal ramp', 'ram-lak']:
        ax[1].plot(w, value, label=key)
                
ax[1].set_title('Ideal ramp filter vs Ram-Lak')
ax[1].legend()

zoom = 5
dw = w[1]-w[0]
zoom_margin = filter_dict['ideal ramp'][ZeroPaddedLength//2 + zoom] * 0.02
ax[1].set_xlim((-1*zoom)*dw, zoom*dw)
ax[1].set_ylim((-1*zoom)*dw * 1/2, zoom*dw)

plt.tight_layout()
plt.show()

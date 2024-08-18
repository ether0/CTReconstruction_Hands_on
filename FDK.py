from Reconstruction_pycuda import Reconstruction
import numpy as np
import os
import logging
import time
import matplotlib.pyplot as plt

pi = np.pi
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

nn = 256
NumberOfDetectorPixels = [512, 384]
NumberOfViews = 90
savedir = 'results'

params = {'SourceInit': [0, 1000.0, 0], 'DetectorInit': [0, -500.0, 0], 'StartAngle': 0, 'EndAngle': 2 * pi,
          'NumberOfDetectorPixels': NumberOfDetectorPixels, 'DetectorPixelSize': [0.5, 0.5], 'NumberOfViews': NumberOfViews,
          'ImagePixelSpacing': [0.5, 0.5, 0.5], 'NumberOfImage': [nn, nn, nn], 'PhantomCenter': [0, 0, 0],
          'RotationOrigin': [0, 0, 0], 'ReconCenter': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'ram-lak',
          'cutoff': 1, 'GPU': 1, 'DetectorShape': 'Flat', 'Pitch': 0, 'DetectorOffset': [0, 0]}


R = Reconstruction(params)

filename = os.path.join('phantoms', f'Shepp_Logan_3d_({nn}x{nn}x{nn}).raw')
R.LoadRecon(filename)

ph = R.image
## FP
start_time = time.time()
R.forward()
log.info(f'Forward {time.time() - start_time:.3f} sec')
proj0 = np.copy(R.proj)
R.SaveProj(os.path.join(savedir, f'proj_SheppLogan_({R.proj.shape[2]}x{R.proj.shape[1]}x{R.proj.shape[0]}).raw'))

## BP
start_time = time.time()
R.backward()
log.info(f'Backward: {time.time() - start_time:.3f} sec')
R.SaveRecon(os.path.join(savedir, f'BP_SheppLogan_({R.image.shape[2]}x{R.image.shape[1]}x{R.image.shape[0]}).raw'))
bp = np.copy(R.image)

## FDK
R.image = np.zeros(params['NumberOfImage'], dtype=np.float32)
start_time = time.time()
R.Filtering()
R.backward()
log.info(f'FDK: {time.time() - start_time:.3f} sec')
R.SaveRecon(os.path.join(savedir, f'Recon_SheppLogan_fdk_({R.image.shape[2]}x{R.image.shape[1]}x{R.image.shape[0]}).raw'))

# Show!
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].imshow(ph[81,:,:], cmap='gray')
ax[0, 0].set_title('Phantom')
ax[0, 1].imshow(proj0[NumberOfViews//2], cmap='gray')
ax[0, 1].set_title('Projection')
ax[1, 0].imshow(bp[81,:,:], cmap='gray')
ax[1, 0].set_title('Backprojection')
ax[1, 1].imshow(R.image[81,:,:], cmap='gray')
ax[1, 1].set_title('Reconstruction')
plt.show()


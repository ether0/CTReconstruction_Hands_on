from Reconstruction_pycuda import Reconstruction
import numpy as np
import glob, sys, os
import logging
import time
import matplotlib.pyplot as plt

pi = np.pi
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

nn = 256
NumberOfDetectorPixels = [512, 384]
NumberOfViews = 90
savedir = 'results'

showImage = True

Niter = 30
alpha = 1

params = {'SourceInit': [0, 1000.0, 0], 'DetectorInit': [0, -500.0, 0], 'StartAngle': 0, 'EndAngle': 2 * pi,
          'NumberOfDetectorPixels': NumberOfDetectorPixels, 'DetectorPixelSize': [0.5, 0.5], 'NumberOfViews': NumberOfViews,
          'ImagePixelSpacing': [0.5, 0.5, 0.5], 'NumberOfImage': [nn, nn, nn], 'PhantomCenter': [0, 0, 0],
          'RotationOrigin': [0, 0, 0], 'ReconCenter': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'hann',
          'cutoff': 1, 'GPU': 1, 'DetectorShape': 'Flat', 'Pitch': 0, 'DetectorOffset': [0, 0]}

R = Reconstruction(params)


projname = os.path.join(savedir, f'proj_SheppLogan_({NumberOfDetectorPixels[0]}x{NumberOfDetectorPixels[1]}x{NumberOfViews}).raw')

filename = os.path.join('phantoms', f'Shepp_Logan_3d_({nn}x{nn}x{nn}).raw')
ph = np.fromfile(filename, dtype=np.float32).reshape([nn, nn, nn])

R.LoadProj(projname)
proj0 = R.proj

eps = 1e-5
norm1 = Reconstruction(params)
norm1.proj = np.ones(
    [params['NumberOfViews'], params['NumberOfDetectorPixels'][1], params['NumberOfDetectorPixels'][0]],
    dtype=np.float32)
norm1.backward()
norm2 = Reconstruction(params)
norm2.image = np.ones(params['NumberOfImage'])
norm2.forward()

rmse = np.zeros(Niter, dtype=np.float32)
start_time = time.time()
for i in range(Niter):
    log.info(f'iter: {i}')
    recon_tmp = R.image
    # proj_fp: projection at the current iteration [40, :, :]
    # image_bp: bakprojection of projection difference [96, :, :]
    # image_upd: final updated image [96, :, :]

    proj_fp = np.zeros([params['NumberOfDetectorPixels'][1], params['NumberOfDetectorPixels'][0]], dtype=np.float32)
    image_bp = np.zeros(params['NumberOfImage'][0:1], dtype=np.float32)
    image_upd = np.zeros(params['NumberOfImage'][0:1], dtype=np.float32)
    #################### Your code here ####################

    ########################################################
    
    rmse[i] = np.sqrt(np.mean((R.image - ph) ** 2))
    log.info(f'RMSE: {rmse[i]}')

    if (showImage == True) and (i % 5 == 0):
        fig, ax = plt.subplots(1,3)  
        ax[0].imshow(proj_fp, cmap='gray')
        ax[1].imshow(image_bp, cmap='gray')
        ax[2].imshow(image_upd, cmap='gray')

        ax[0].set_title('Current Projection')
        ax[1].set_title('Difference Backprojection')
        ax[2].set_title('Updated Image')
        fig.suptitle(f'{i+1}/{Niter} iter', fontsize=16)

        for a in ax:
            a.axis('off')

        plt.tight_layout()
        plt.show()

log.info(f'SART: {time.time() - start_time:.3f} sec')
R.SaveRecon(os.path.join(savedir, f'Recon_SheppLogan_sart_iter{Niter}_({R.image.shape[2]}x{R.image.shape[1]}x{R.image.shape[0]}).raw'))

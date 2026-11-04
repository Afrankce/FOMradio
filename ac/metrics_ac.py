"""Native-grid, valid-support metrics for exploratory U6G evaluation."""
import math
import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion,gaussian_filter


def boundary_from_height(height):
    """Conservative 2x2 occupancy pooling, followed by 3-pixel dilation.

    Published height maps are 256x256 and radio maps are 128x128. A native
    radio-map cell is occupied if any of its four height cells is occupied.
    """
    h=np.asarray(height)
    if h.shape!=(256,256) or not np.isfinite(h).all():raise ValueError('bad height')
    occupied=(h>0).reshape(128,2,128,2).any(axis=(1,3))
    return binary_dilation(occupied,structure=np.ones((7,7),bool))&~occupied


def score_map(prediction,ground_truth,valid,boundary):
    """Per-image MAE, RMSE, PSNR, masked SSIM and building-boundary MAE.

    PSNR/SSIM data range is 300 dB; Gaussian SSIM uses raw dB coordinates,
    sigma=1.5, radius=5, K1=.01, K2=.03, and population covariance.
    Only windows entirely inside the image and valid support are counted.
    No output clipping or per-image normalization is performed.
    """
    y=np.asarray(ground_truth,dtype=np.float64)
    x=np.asarray(prediction,dtype=np.float64)
    valid=np.asarray(valid,dtype=bool);boundary=np.asarray(boundary,dtype=bool)
    if x.shape!=y.shape or valid.shape!=y.shape or boundary.shape!=y.shape or y.ndim!=2:
        raise ValueError('incompatible metric shapes')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('non-finite map')
    if not valid.any():raise ValueError('no valid target pixels')
    error=x[valid]-y[valid];mse=float(np.mean(np.square(error)));mae=float(np.mean(np.abs(error)))
    boundary_valid=valid&boundary
    eroded=binary_erosion(valid,structure=np.ones((11,11),bool),border_value=0)
    ssim=None
    if eroded.any():
        def g(a):return gaussian_filter(a,sigma=1.5,truncate=3.5,mode='reflect')
        ux,uy=g(x),g(y)
        vx=g(x*x)-ux*ux;vy=g(y*y)-uy*uy;vxy=g(x*y)-ux*uy
        c1=(.01*300.)**2;c2=(.03*300.)**2
        den=(ux*ux+uy*uy+c1)*(vx+vy+c2)
        value=((2*ux*uy+c1)*(2*vxy+c2))/den
        if not np.isfinite(value[eroded]).all():raise ValueError('nonfinite valid SSIM')
        ssim=float(value[eroded].mean())
    return {'mae_db':mae,'rmse_db':math.sqrt(mse),
            'psnr_db':10*math.log10(300.**2/mse) if mse>0 else None,
            'psnr_infinite':mse==0.,'ssim':ssim,'valid_ssim_windows':int(eroded.sum()),
            'boundary_mae_db':float(np.abs(x[boundary_valid]-y[boundary_valid]).mean()) if boundary_valid.any() else None,
            'boundary_valid_pixels':int(boundary_valid.sum()),'valid_pixels':int(valid.sum()),
            'outside_range_fraction':float(((x< -300.)|(x>0.)).mean())}

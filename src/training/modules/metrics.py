import json
import numpy as np
from .util_data import binary_threshold

def binary_metrics(_vol_t, _vol_s, threshold_1=0.04, dec_points = 5):
    
    if _vol_t.shape != _vol_t.shape:
        raise ValueError ("Shape mismatch") 
    vol_t = np.squeeze(binary_threshold(_vol_t, threshold=threshold_1))
    vol_s = np.squeeze(binary_threshold(_vol_s, threshold=threshold_1))
    intersection= (np.logical_and(vol_s, vol_t)).sum()
    union = (vol_s.sum() + vol_t.sum())
    if intersection ==0 and union == 0:
        return 1., 1., 1., 1.
    if union == 0:
        union = 1e-7
    dice_value = np.round(2. * intersection / union, dec_points)
    ppv = np.round(intersection / vol_t.sum(), dec_points)
    sensitivity = np.round(intersection / vol_s.sum(), dec_points)
    volume_ratio = np.round(vol_t.sum() / vol_s.sum(), dec_points)
    return dice_value, ppv, sensitivity, volume_ratio

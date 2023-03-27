import numpy as np
import tensorflow as tf

from scipy import ndimage
from functools import partial

from skimage.morphology import remove_small_objects
from skimage.morphology import remove_small_holes



def get_lat_var(batch_size, z_dim):
    z = tf.random.normal((batch_size, 1, 1, z_dim))
    return z 

def cGAN_input_gen(input_image,
                   n_z,
                   z_dim,
                   zero_z=False):
    
    glv = partial(get_lat_var, z_dim=z_dim)    
    z = glv(batch_size=n_z) 
    
    if zero_z:
        z = tf.zeros_like(z)

    input_image = tf.constant(input_image[None,:,:,None])
    cGAN_input  = tf.repeat(input_image, n_z, axis=0)
    
    return cGAN_input, z  
    
def cGAN_predict_single_image(G_model, input_image, n_z, z_dim, verbose=False):
    
    sample_preds = []
    
    cGAN_input, z = cGAN_input_gen(input_image, n_z, z_dim, zero_z=False)
#     _sample_stat_y = stat_y[_samples*mc_samples : (_samples+1)*mc_samples]
#     _sample_stat_z = stat_z[_samples*mc_samples : (_samples+1)*mc_samples]
    
    try:
        sample_pred = G_model([cGAN_input, z], training=None).numpy()
        if verbose:
            print("inference using device: GPU")
#             print("_sample_pred_", np.shape(_sample_pred_))
        sample_preds += [sample_pred]
    
    except:
        with tf.device('/device:cpu:0'):
            sample_pred = G_model([cGAN_input, z], training=None).numpy()
            if verbose:
                print("inference using device: CPU")
#                 print("_sample_pred_", np.shape(_sample_pred_))
            sample_preds += [sample_pred]
#     predictions = np.reshape(predictions, (slices_number, mc_samples, x_W, x_H, x_C))
    sample_preds = np.squeeze(np.array(sample_preds))
    if n_z == 1:
        sample_preds = sample_preds[None,:,:]
    return sample_preds

def cGAN_predict_volume(G_model, input_volume, n_z, z_dim, post=True, threshold = 0.04, smooth_size = 3):
    output_volume_mean = []
    output_volume_std  = []
    no_slices = np.shape(input_volume)[0]
    for i in range(no_slices):
        input_image = input_volume[i]
        output_samples = cGAN_predict_single_image(G_model,
                                                   input_image,
                                                   n_z=n_z,
                                                   z_dim=z_dim
                                                  )                                                  
#                                                    x_W=x_W, x_H=x_H, z_dim=z_dim)
        output_image_mean = np.squeeze(np.mean(output_samples, axis = 0))
        output_image_std  = np.squeeze(np.std(output_samples, axis = 0))
        output_volume_mean += [output_image_mean]
        output_volume_std  += [output_image_std]
    
    output_volume_mean = np.array(output_volume_mean)
    output_volume_std  = np.array(output_volume_std)
    
    output_volume_mean_mask = binary_threshold(output_volume_mean,  threshold=threshold)
    
    if post:
        
        output_volume_mean_mask = post_process(output_volume_mean_mask, smooth_size = 3)
        
    output_volume_mean = output_volume_mean * output_volume_mean_mask
    
    return output_volume_mean, output_volume_std, output_volume_mean_mask

def binary_threshold(volume, threshold=0.04, dtype=np.float32):
    volume_cp = np.copy(volume)
    volume_cp [volume_cp<threshold] = 0
    volume_cp [volume_cp>=threshold] = 1
    return np.array(volume_cp).astype(dtype)

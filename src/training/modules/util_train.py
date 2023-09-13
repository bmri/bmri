import os
import argparse, textwrap
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import ndimage
from functools import partial
from skimage.transform import resize
from util_data import binary_threshold
from skimage.morphology import remove_small_objects
from skimage.morphology import remove_small_holes

formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=50)

def args_params():
    
    parser = argparse.ArgumentParser(description='list of arguments',formatter_class=formatter)
    parser.add_argument('--device'   , type=int, default=0, help=textwrap.dedent('''GPU id to use'''))
    parser.add_argument('--datadir', type=str, default="./data/", required=True, help=textwrap.dedent('''training and validation data directory'''))
    parser.add_argument('--gp_coef'   , type=float, default=10.0, help=textwrap.dedent('''Gradient penalty coefficient'''))
    parser.add_argument('--n_critic'  , type=int, default=4, help=textwrap.dedent('''Number of critic updates per generator update'''))
    parser.add_argument('--n_epoch'   , type=int, default=1000, help=textwrap.dedent('''Maximum number of epochs'''))
    parser.add_argument('--z_dim'     , type=int, default=128, help=textwrap.dedent('''Dimension of the latent variable'''))
    parser.add_argument('--batch_size', type=int, default=16, help=textwrap.dedent('''Training batch'''))
    parser.add_argument('--seed_no'      , type=int, default=0, help=textwrap.dedent('''Set random seed'''))
    parser.add_argument('--savefig_freq' , type=int, default=200, help=textwrap.dedent('''Number of batches before saving plots'''))
    parser.add_argument('--save_suffix'  , type=str, default='brain_extraction_01', help=textwrap.dedent('''Network/results directory suffix'''))
    parser.add_argument('--n_z_stat'   , type=int, default=20, help=textwrap.dedent('''Number of samples'''))
    parser.add_argument('--sample_plots' , type=int, default=12, help=textwrap.dedent('''Number of validation images to generate plots'''))
    parser.add_argument('--main_dir'     , type=str, default='./results/', help=textwrap.dedent('''Results parent directory'''))

    return parser.parse_args()

def normalize(data, scale=1., data_type = np.float32):
    """
    returns a min-max normalized data as numpy array 
    default scaling is one but can be passed as arg
    """
    if np.max(data) == 0 and np.min(data) == 0:
        return data
    else:
        return scale*((np.array(data) - np.min(data)) / (np.max(data) - np.min(data))).astype(data_type)
    
def get_lat_var(batch_size, z_dim):
    z = tf.random.normal((batch_size,1,1,z_dim))
    return z 

def save_loss(loss,loss_name,savedir,n_epoch):    

    np.savetxt(f"{savedir}/{loss_name}.txt",loss)
    fig, ax1 = plt.subplots()
    ax1.plot(loss)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel(loss_name)
    ax1.set_xlim([1,len(loss)])

    ax2 = ax1.twiny()
    ax2.set_xlim([0,n_epoch])
    ax2.set_xlabel('Epochs')

    plt.tight_layout()
    plt.savefig(f"{savedir}/{loss_name}.png", dpi=200)    
    plt.close()  
    
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

    try:
        sample_pred = G_model([cGAN_input, z], training=None).numpy()
        if verbose:
            print("inference using device: GPU")
        sample_preds += [sample_pred]
    
    except:
        with tf.device('/device:cpu:0'):
            sample_pred = G_model([cGAN_input, z], training=None).numpy()
            if verbose:
                print("inference using device: CPU")
            sample_preds += [sample_pred]
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

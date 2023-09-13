import os
import h5py
import numpy as np
import tensorflow as tf

def binary_threshold(volume, threshold=0.04, dtype=np.float32):
    volume_cp = np.copy(volume)
    volume_cp [volume_cp<threshold] = 0
    volume_cp [volume_cp>=threshold] = 1
    return np.array(volume_cp).astype(dtype)

def read_hdf5(
    filename_list,
    batch_size,
    load_images=None,
    drop_remainder=True,
    down_sample=2,
    down_size_1=1,
    down_size_2=1,
    modalities=[1,0],
    dataset=False):

    for filename in filename_list:
        assert os.path.exists(filename), 'Error: The data file ' + filename + ' is unavailable.'
        file = h5py.File(filename, "r+")
        img  = file["/images"][::down_sample, ::down_size_1, ::down_size_2, :]
        print(f'Loading ... {len(img)} images from {filename}')
        if load_images is None:
            load_images = img
        else:
            load_images=np.concatenate((load_images, img), axis=0)      
        file.close()
       
    load_images[...,[0,1]] = load_images[...,[1,0]]
    # print("states:", np.max(load_images), np.mean(load_images), "shape:", load_images[0].shape)
    height, width, chanels = load_images[0].shape
    chanels = int(chanels//2)
    N_train      = len(load_images)
    if dataset:
        train_data = tf.data.Dataset.from_tensor_slices(load_images).shuffle(N_train).batch(batch_size, drop_remainder=True)
        print("return training data as tf.data.Dataset")
    else:
        train_data = load_images
        print("return training data as matrix")

    return load_images, height, width, chanels

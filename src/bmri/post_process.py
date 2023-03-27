import numpy as np
from scipy.ndimage import median_filter

from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import sobel as sobel_edge


def cGAN_bet_pp(be_mean, threshold=0.1):
    out = np.zeros_like(be_mean)
    for i in range(np.shape(be_mean)[0]):
        image = be_mean[i] 
        image = image + 2 * sobel_edge(image)
        # image = skimage.filters.median(image, )
        block_size = 5
        local_thresh = threshold_local(image,
                                       block_size,
                                       offset=0,
                                       method = "mean") # method : {'generic', 'gaussian', 'mean', 'median'}
        binary_local = image > local_thresh

        binary_local = remove_small_objects(binary_local)
        binary_local = remove_small_holes(binary_local)

        binary = binary_threshold(image, threshold=threshold)
        binary_local = (binary_local + binary).astype(np.bool)

#         binary_local = skimage.morphology.remove_small_objects(binary_local, 200)
#         binary_local = skimage.morphology.remove_small_holes(binary_local, 200)
        

        out[i] = binary_local.astype(np.bool)
    out = median_filter (out, 5) 
    
    binary_local = remove_small_objects(binary_local, 1000)
    binary_local = remove_small_holes(binary_local, 1000)
        
    label_img = label(out)

    r = regionprops(label_img) # l is from previous approach
    out = (label_img==(1+np.argmax([i.area for i in r]))).astype(int)
#     out = skimage.morphology.dilation(out)

    return out

def binary_threshold(volume, threshold=0.04, dtype=np.float32):
    volume_cp = np.copy(volume)
    volume_cp [volume_cp<threshold] = 0
    volume_cp [volume_cp>=threshold] = 1
    return np.array(volume_cp).astype(dtype)

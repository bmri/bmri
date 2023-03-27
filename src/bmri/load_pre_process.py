import nibabel as nib
import numpy as np

from skimage.transform import resize


def canonical_to_axial(img_canonical_data, nose_up = True):
    img_axial = np.swapaxes(img_canonical_data, 0, 2)
    if nose_up: img_axial = np.flip(img_axial, axis = 1) # nose upward in plt.imshow()
    img_axial = np.squeeze(img_axial)
    return img_axial

def axial_to_canonical(img_axial_data, nose_up = True):
    
    if nose_up: img_axial_data = np.flip(img_axial_data, axis = 1) # nose upward in plt.imshow()

    img_canonical_data = np.swapaxes(img_axial_data, 0, 2)

    return img_canonical_data


def img_to_dl_input_axial(img_axial, model_H, model_W, pad1, pad2, slices=None, add_dim = True):
    """
    input image data should be in axial orientation
    """
    assert len(img_axial.shape) == 3, "image dimension should be 3"
    original_size = img_axial.shape 
    if slices==None: slices = original_size[0]
    
    window_shape=(slices, model_H-2*pad1, model_W-2*pad2)  
#     print("window_shape", window_shape)
    img_window = resize(img_axial, output_shape=window_shape, order=1, anti_aliasing=True, preserve_range=True)

    img_resized = np.zeros(shape = (slices, model_H, model_W))
    img_resized[:,pad1:model_H-pad1,pad2:model_W-pad2] = img_window
    
    if add_dim: img_resized = img_resized[...,np.newaxis]
        
    return img_resized

def dl_output_to_axial(dl_out_img, model_H, model_W, pad1, pad2, original_axial_shape):

    dl_out_img_cropped = dl_out_img[:,pad1:model_H-pad1,pad2:model_W-pad2,:]
    img_axial_org = resize(dl_out_img_cropped, output_shape=original_axial_shape, order=1, anti_aliasing=True, preserve_range=True)
    return img_axial_org
    
    
# if slice_normalized:  
#     t1_img = np.array([normalize(t1_img[slice_index]) for slice_index in range(len(t1_img))])
    
def normalize(data, scale=1., data_type = np.float32):
    """
    returns a min-max normalized data as numpy array 
    default scaling is one but can be passed as arg
    """
    if np.max(data) == 0 and np.min(data) == 0:
        return data
    else:
        return scale*((np.array(data) - np.min(data)) / (np.max(data) - np.min(data))).astype(data_type)    

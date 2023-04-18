import os
import sys
# print(sys.executable)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import tensorflow as tf
import numpy as np

import nibabel as nib

from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
from skimage.transform import resize
from pathlib import Path

from load_pre_process import *
from models import generator_BE
from cGANs import cGAN_predict_volume
from post_process import cGAN_bet_pp
from utils import saved_models_dir, sample_images_dir, outputs_dir
from utils import check_BE_model
# from utils import save_report_images

os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description="Process Nifti files.")

parser.add_argument("-i", "--input", dest="input_head_img", type=str, required=True,
                    help="Input 3D T1 head MRI image")

parser.add_argument("-o", "--output", dest="output_brain_img", type=str, required=True,
                    help="Output extracted brain image")

parser.add_argument("-n", "--num-samples", dest="num_samples", type=int, default=10,
                    help="The number of sample brains to generate")

# parser.add_argument("-r", "--report", dest="report_dir", type=str, default=outputs_dir,
#                     help="The directory to save the repoty")

args = parser.parse_args()

# Parameters
model_H, model_W, pad1, pad2 = 128, 128, 4, 4
nose_up = True
z_dim = 128
model_dir = saved_models_dir

n_samples   = int(args.num_samples)

input_path = Path(args.input_head_img)
input_file_name = input_path.name



# loading the model
check_BE_model()

BE_model = generator_BE(Input_W=128,
                        Input_H=128,
                        Input_C=1,
                        Z_Dim=128,
                        k0=16,
                        reg_param=1e-07)

# BE_model.load_weights(f'{model_dir}/model_brain_extraction_cGAN_128x128_z128_weights.h5')
BE_model.load_weights(f'{model_dir}/model_brain_extraction_cGAN_128x128_z128_weights_dl.h5')
print("Model loaded successfully")
# Loading the image
img = nib.load(args.input_head_img)
image_data_shape = np.shape(img.get_fdata())
print(f"Image original size: {image_data_shape}")

# This part is inspired from: https://github.com/nipy/nibabel/issues/1063#issuecomment-967124057
# keep original orientation
img_ornt = io_orientation(img.affine)
ras_ornt = axcodes2ornt("RAS")

to_canonical = img_ornt  # Same as ornt_transform(img_ornt, ras_ornt)
from_canonical = ornt_transform(ras_ornt, img_ornt)

# Same as as_closest_canonical
img_canonical = img.as_reoriented(to_canonical)
img_canonical_data = img_canonical.get_fdata()


# canonical_to_axial
img_axial = canonical_to_axial(img_canonical_data, nose_up=nose_up)
img_axial_shape = np.shape(img_axial)

# axial image to deep learning model input (paddig and resizing)
img_axial_dl_input = img_to_dl_input_axial(img_axial, model_H, model_W, pad1, pad2)

# Normalize the data (this version is volume-based normalized)
img_axial_dl_input = normalize(img_axial_dl_input)

# img_transformed = dummy(img_axial_dl_input) # to test other functions

# Perfomr prediction
be_mean, be_std, be_mask = cGAN_predict_volume(G_model = BE_model,
                                               input_volume = img_axial_dl_input,
                                               n_z   = n_samples,
                                               z_dim = z_dim,
                                               post=False,
                                               threshold = 0.01)

be_mask_pp = cGAN_bet_pp(be_mean)[:,:,:,np.newaxis]

# @ TODO:
# Generating report 
# sample_img = np.flip(np.squeeze(img_axial_dl_input)[:,:,64])
# sample_be  = np.flip((np.squeeze(be_mask_pp)*np.squeeze(be_mean))[:,:,64])
# sample_std = np.flip(np.squeeze(be_std)[:,:,64])

# try:
#     save_report_images([sample_img, sample_be, sample_std], report_dir=args.report_dir)
# except:
#     print("Could not generate report")
#     pass

# img_transformed back to axial original size:
be_mask_pp_resized = dl_output_to_axial(be_mask_pp, model_H, model_W, pad1, pad2, original_axial_shape = img_axial_shape)

# axial to canonical
be_mask_pp_canonical_data =  axial_to_canonical(be_mask_pp_resized, nose_up=nose_up)
be_mask_pp_canonical_data = np.reshape(be_mask_pp_canonical_data, np.shape(img_canonical_data))

img_transformed_canonical_data = be_mask_pp_canonical_data * img_canonical_data

# These are for dummy function devepment, ignore them
# print(np.shape(img_canonical_data))
# print(np.shape(img_transformed_canonical_data))
# img_original = img_canonical.as_reoriented(from_canonical)
# img_transformed = img_canonical

img_transformed = nib.Nifti1Image(img_transformed_canonical_data, img_canonical.affine, img_canonical.header)
img_transformed_original = img_transformed.as_reoriented(from_canonical)


save_address = args.output_brain_img
try:
    nib.save(img_transformed_original, save_address)
except:
    print(f"Could not save in the provided save address: Check {outputs_dir} instead")
    save_address = input_file_name.split(".nii.gz")[0] + "cGAN-bet_brain" + ".nii.gz"
    nib.save(img_transformed_original, outputs_dir + save_address )
    
    


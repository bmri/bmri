import os
import gdown

saved_models_dir  = os.path.join(os.path.expanduser('~'), 'cGAN-BET/saved_models')
sample_images_dir = os.path.join(os.path.expanduser('~'), 'cGAN-BET/sample_images')
outputs_dir       = os.path.join(os.path.expanduser('~'), 'cGAN-BET/outputs_dir/')


import matplotlib.pyplot as plt

def save_report_images(images, report_dir=outputs_dir):
    img_name = ["sample_img", "sample_be", "sample_std"]
    cmaps = ["cividis", "cividis", "jet"]
    for i, img in enumerate(images):
        fig, ax = plt.subplots()
        plt.axis("off")
        ax.imshow(img, cmap=cmaps[i])
        plt.savefig(f'{report_dir}/{img_name[i]}.png')
        


def check_BE_model():
#     from bmri.utils import saved_models_dir, sample_images_dir, outputs_dir # if run from outside

    model_url  = 'https://drive.google.com/uc?export=download&id=1EzxOFjCJO5adRaUbJKf7aJKvvW1L41JN'    
    model_name = 'model_brain_extraction_cGAN_128x128_z128_weights_dl.h5'
    
    sample_image_url  = 'https://drive.google.com/uc?export=download&id=1F8I6WmhrwOhpGoRyXQIaY4jAibmjMOC9'    
    sample_image_name = 'CC359_CC0001_philips_15_55_M.nii.gz'


    def create_dirs(_dir):
        if not os.path.exists(_dir):
            print(f" directory does not exist. Create: {_dir}")
            os.makedirs(_dir)

    for _dir in [saved_models_dir, sample_images_dir, outputs_dir]: create_dirs(_dir)  

    # Check if file exists and download if not
    if not os.path.isfile(saved_models_dir+"/"+model_name):
        url = model_url
        output = saved_models_dir+"/"+model_name
        gdown.download(url, output, quiet=False)
        
    if not os.path.isfile(sample_images_dir+"/"+sample_image_name):
        url = sample_image_url
        output = sample_images_dir+"/"+sample_image_name
        gdown.download(url, output, quiet=False)
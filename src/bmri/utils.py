import os

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
        



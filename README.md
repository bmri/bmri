# bmri: cGAN-BET BRAIN EXTRACTION

## Description

This version of the bmri repository is provided for the reviewers' consideration for the following paper:

Probabilistic Brain Extraction in MR Images via Conditional Generative Adversarial Networks

https://www.biorxiv.org/content/10.1101/2022.03.14.484346v1

An updated version of the paper is under review.
Please feel free to contact if you have any questions or feedback.

## Installation
It is highly recommended to use a conda environment for installation.

```bash
conda create -n bmri_env python=3.9
conda activate bmri_env
```
We also recommend installing tensorflow-gpu 2.4.1 using: 
```bash
conda install -c anaconda tensorflow-gpu=2.4.1
```
It is also optional but recommended to install bmri package in the home directory as:
```bash
cd
mkdir cGAN-BET
cd cGAN-BET
```
Then install the package as:
```bash
git clone https://github.com/bmri/bmri.git
cd bmri
pip install .
```


## Usage (Inference)
Please make sure to follow installation instructions and have the environment activated

To use brain extraction find "cGAN-bet.py" in the "bmri_dir/bmri/src/bmri/"

To perform brain extraction, run the python code as:

```bash
python ~/cGAN-BET/bmri/src/bmri/cGAN-bet.py -i ~/cGAN-BET/sample_images/sample_cc359_0001.nii.gz -o ~/cGAN-BET/outputs_dir/sample_cc359_0001_cGAN_bet_brain.nii.gz
```
The number of samples can also be provided using optional flag -n (default is 20 use lower values if no GPU is available)

In the first run, the script will automatically download the model.

Also it downloads a sample image from cc359 dataset (https://www.ccdataset.com/) for testing the algorithm to: ~/cGAN-BET/sample_images/

It also creates an output directory. If the image can not be saved as the provided path_to_output_brain_image.nii.gz, it will be saved in ~/cGAN-BET/outputs_dir/ to prevent losing the result.

## Usage (Training)
Please refer to https://github.com/bmri/bmri/tree/main/src/training for training a cGAN model

## Citation
Please cite the following paper if you use cGAN-bet
https://www.biorxiv.org/content/10.1101/2022.03.14.484346v1
An updated version of the paper is under review. The references will be updated accordingly.


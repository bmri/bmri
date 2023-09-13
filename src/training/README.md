
## usuag:
The following provides the instructions and parameters for training.
``` bash
python train_cGAN.py -h
```
``` bash
usage: train_cGAN.py [-h] [--device DEVICE] --datadir DATADIR [--gp_coef GP_COEF] [--n_critic N_CRITIC] [--n_epoch N_EPOCH]
                     [--z_dim Z_DIM] [--batch_size BATCH_SIZE] [--seed_no SEED_NO] [--savefig_freq SAVEFIG_FREQ]
                     [--save_suffix SAVE_SUFFIX] [--n_z_stat N_Z_STAT] [--sample_plots SAMPLE_PLOTS] [--main_dir MAIN_DIR]

list of arguments

optional arguments:
  -h, --help                   show this help message and exit
  --device DEVICE              GPU id to use
  --datadir DATADIR            training and validation data directory
  --gp_coef GP_COEF            Gradient penalty coefficient
  --n_critic N_CRITIC          Number of critic updates per generator update
  --n_epoch N_EPOCH            Maximum number of epochs
  --z_dim Z_DIM                Dimension of the latent variable
  --batch_size BATCH_SIZE      Training batch
  --seed_no SEED_NO            Set random seed
  --savefig_freq SAVEFIG_FREQ  Number of batches before saving plots
  --save_suffix SAVE_SUFFIX    Network/results directory suffix
  --n_z_stat N_Z_STAT          Number of samples
  --sample_plots SAMPLE_PLOTS  Number of validation images to generate plots
  --main_dir MAIN_DIR          Results parent directory
```
datasets should be strored in .h5 format insidet DATADIR/train/.h5  and DATADIR/valid/.h5

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], './modules'))

import numpy as np
from tqdm import tqdm
from glob import glob
from time import time, sleep

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from modules.util_train import *
from modules.util_data import read_hdf5, binary_threshold
from modules.model_train import generator, critic, gradient_penalty
from modules.metrics import binary_metrics

PARAMS = args_params()
os.environ["CUDA_VISIBLE_DEVICES"]=str(PARAMS.device)
# os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '16200'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')
# print ("Setting TensorFlow verbosity to silent! ...")

n_epoch     = PARAMS.n_epoch
n_critic    = PARAMS.n_critic
z_dim       = PARAMS.z_dim
batch_size  = PARAMS.batch_size
gp_coef     = PARAMS.gp_coef
# reg_param   = PARAMS.reg_param
save_suffix = PARAMS.save_suffix
datadir     = PARAMS.datadir
n_z_stat    = PARAMS.n_z_stat
n_out       = PARAMS.sample_plots
savefig_freq    = PARAMS.savefig_freq

save_model_freq = 5 * savefig_freq
verbose_iter = savefig_freq
down_size   = 1
inversed = False
if inversed: modalities[0,1]=modalities[1,0]
train_inc = 2
valid_inc = 50
modalities = [1, 0]
description ="""
Brain extraction log.

"""
savedir = f"results/{save_suffix}_BS{batch_size}_"
print(f'Creating output dir:\t {savedir}\n')
if not os.path.exists(savedir):
    os.makedirs(savedir) 
    # print(f"folder {savedir} created...")
# else:
    # print('\n Folder already exists!\n')    

# print('\n Saving parameters to file \n')

description_file = savedir + '/description.txt'

print(f'Saving params. to file:\t {description_file}\n')

try:
    with open(description_file,"w") as _f:    
        _f.write(description)  
        for _param in vars(PARAMS):
            _f.write(f"{_param} = {vars(PARAMS)[_param]}\n")          
except:
    print("ERROR creating description file")

print('Loading data from files ________________\n')



train_file_list = np.sort(glob(datadir+"/train*/*.h5"))
valid_file_list = np.sort(glob(datadir+"/valid*/*.h5"))

train_data, x_W, x_H, x_C = read_hdf5(
    filename_list=train_file_list,
    batch_size=batch_size,
    drop_remainder=True,
    down_sample=train_inc,
    down_size_1=down_size,
    down_size_2=down_size,
    modalities=modalities,
    dataset=False
              ) 

valid_data, _, _, _ = read_hdf5(
    filename_list=valid_file_list,
    batch_size=batch_size,
    drop_remainder=True,
    down_sample=valid_inc,
    down_size_1=down_size,
    down_size_2=down_size,
    modalities=modalities,
    dataset=False
              ) 

train_data_shape = train_data.shape
rnd_indeces = np.arange(train_data_shape[0])
np.random.shuffle(rnd_indeces)

train_data = train_data[rnd_indeces]

for i in range(len(train_data)):
    train_data[i,:,:,0] = normalize(train_data[i,:,:,0])
    train_data[i,:,:,1] = normalize(train_data[i,:,:,1])

train_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(np.shape(train_data)[0]).batch(batch_size, drop_remainder=True)

print('Creatinmg cGAN models  ________________\n')
G_model = generator(
    Input_W=x_W, Input_H=x_H, Input_C=x_C,
    Z_Dim=z_dim,
    k0=16
    )
    
D_model = critic(
    Input_W=x_W,Input_H=x_H,Input_C=x_C,
    k0=16
    )

G_optim = tf.keras.optimizers.Adam(0.0001,beta_1=0.2, beta_2=0.7, amsgrad=True)
D_optim = tf.keras.optimizers.Adam(0.0001,beta_1=0.2, beta_2=0.7, amsgrad=True)

glv = partial(get_lat_var,
              z_dim=z_dim)

stat_z   = glv(batch_size=n_z_stat*n_out)

valid_x = valid_data[0:n_out][:,:,:,0:x_C]
valid_y = tf.constant(valid_data[0:n_out])[:,:,:,x_C::]

stat_y  = tf.repeat(valid_y, n_z_stat, axis=0)

n_iters = 1
G_loss_log    = []
D_loss_log    = []
wd_loss_log   = []
dice_metric   = []

brain_mask_img = binary_threshold(valid_data[:,:,:,0])
valid_input  = valid_data[:,:,:,1]

print("train_data, valid_data, N_valid, x_W, x_H, x_C")
print("shapes", train_data,
      train_data_shape,
      np.shape(valid_data),
      x_W, x_H, x_C)

print('Started training  ________________\n')
for i in range(n_epoch):
    print(f"epoch: {i}")

    for n_batch, true in tqdm(enumerate(train_data)):

        true_X = true[:,:,:,0:x_C]  
        true_Y = true[:,:,:,x_C::]  

        z = glv(batch_size=batch_size)

        with tf.GradientTape() as tape:
            fake_X      = G_model([true_Y, z], training=True)
            fake        = tf.squeeze(tf.concat([fake_X, true_Y], axis=3))
            fake_XY_val = D_model(fake, training=True)
            true_XY_val = D_model(true, training=True)

            gp = gradient_penalty(fake_X, true_X, true_Y,D_model, p=2)

            fake_loss = tf.reduce_mean(fake_XY_val)
            true_loss = tf.reduce_mean(true_XY_val) 
            wd_loss = true_loss - fake_loss                
            D_loss = -wd_loss + gp_coef*gp              

        D_gradient = tape.gradient(D_loss, D_model.trainable_variables)
        D_optim.apply_gradients(zip(D_gradient, D_model.trainable_variables))

        D_loss_log.append(D_loss.numpy())
        wd_loss_log.append(wd_loss.numpy())

        del tape

        if (n_iters) % (n_critic) == 0:

            with tf.GradientTape() as tape:
                fake_X      = G_model([true_Y,z],training=True)
                fake        = tf.squeeze(tf.concat([fake_X,true_Y],axis=3))
                fake_XY_val = D_model(fake,training=True)

                G_loss = -tf.reduce_mean(fake_XY_val)

            gen_gradient = tape.gradient(G_loss, G_model.trainable_variables)
            G_optim.apply_gradients(zip(gen_gradient, G_model.trainable_variables))

            G_loss_log.append(G_loss.numpy())

            del tape

        n_iters += 1    

        if ((n_batch+1) % verbose_iter) == 0:
            print(f" *** iter:{n_iters} ---> d_loss:{D_loss.numpy():.4e}, gp_term:{gp.numpy():.4e}, wd:{wd_loss.numpy():.4e}")

        if ((n_batch+1) % savefig_freq == 0):
            print("validation ... ")
            start_time = time()
    
            be_mean, be_std, be_mask = cGAN_predict_volume(G_model = G_model,
                                                                input_volume = valid_input,
                                                                n_z = 4,
                                                                z_dim = z_dim,
                                                                post=False,
                                                                threshold=0.01,
                                                                smooth_size=3)

            be_mask_smoothed = binary_threshold(be_mean, 0.1)

            volume_metrics = binary_metrics(be_mask_smoothed, brain_mask_img)    
            print("validation metrics:", *volume_metrics) 
            dice_metric.append(volume_metrics[0])
            print(f"{i}: Validation step Run time: {time() - start_time}") 

        if ((n_batch+1) % savefig_freq == 0):

            print("     *** Saving plots") 

            predictions = []
            for _samples in range(n_out):
                _sample_stat_y = stat_y[_samples*n_z_stat : (_samples+1)*n_z_stat]
                _sample_stat_z = stat_z[_samples*n_z_stat : (_samples+1)*n_z_stat]

                try:
                    _sample_pred_ = G_model([_sample_stat_y, _sample_stat_z], training=None).numpy()
                    predictions += [_sample_pred_]

                except:
                    with tf.device('/device:cpu:0'):
                        _sample_pred_ = G_model([_sample_stat_y, _sample_stat_z], training=None).numpy()
                        print("inference using device: CPU")
                        predictions += [_sample_pred_]

            pred_ = np.reshape(predictions, (n_z_stat*n_out, x_W, x_H, x_C))

            ncol = 4
            fig1, axs1 = plt.subplots(n_out, ncol, dpi=72, figsize=(ncol*5,n_out*5))
            ax1 = axs1.flatten()
            ax_ind = 0

            for t in range(n_out):
                axs = ax1[ax_ind]
                pcm = axs.imshow(valid_y[t].numpy(),aspect='equal', cmap='cividis')
                # plt.colorbar(pcm)
                # plt.show()
                if t==0:
                    axs.set_title(f'measurement', fontsize=30)
                axs.axis('off')
                ax_ind +=1
                axs = ax1[ax_ind]
                pcm = axs.imshow(valid_x[t],aspect='equal', cmap='cividis')
                # plt.colorbar(pcm)
                # plt.show()
                if t==0:
                    axs.set_title(f'target',fontsize=30)
                axs.axis('off')
                ax_ind +=1
                sample_mean = tf.math.reduce_mean(pred_[t*n_z_stat:(t+1)*n_z_stat],axis=0).numpy()
                sample_std = tf.math.reduce_std(pred_[t*n_z_stat:(t+1)*n_z_stat],axis=0).numpy()
                axs = ax1[ax_ind]
                pcm = axs.imshow(sample_mean,aspect='equal', cmap='cividis')
                # plt.colorbar(pcm)
                # plt.show()
                if t==0:
                    axs.set_title(f'mean',fontsize=30)
                axs.axis('off')
                ax_ind +=1
                axs = ax1[ax_ind]
                pcm = axs.imshow(sample_std,aspect='equal', cmap='jet')
                # plt.colorbar(pcm)
                # plt.show()
                if t==0:
                    axs.set_title(f'std',fontsize=30)
                axs.axis('off')
                ax_ind +=1
            fig1.tight_layout()
            fig1.savefig(f"{savedir}/sample_stats_{i+1}_{n_batch+1}.png", format = 'png')
            plt.close('all')

        if ((n_batch+1) % save_model_freq == 0):
            print(" Saving model")
            G_model.save(f'{savedir}/models/G_model_{i+1}_{n_batch+1}'+'.h5')
            D_model.save(f'{savedir}/models/D_model_{i+1}_{n_batch+1}'+'.h5')
            save_loss(G_loss_log,'g_loss',savedir,n_epoch)
            save_loss(D_loss_log,'d_loss',savedir,n_epoch)  
            save_loss(wd_loss_log,'wd_loss',savedir,n_epoch) 
            save_loss(dice_metric,'dice_metric',savedir,n_epoch)

print('Training Completed ________________\n')

import os
import sys
import tensorflow as tf

from bmri import metrics
from bmri.utils import saved_models_dir, sample_images_dir

# cwd = os.getcwd()
# sys.path.insert(1, os.path.join(sys.path[0], cwd+'/home/saeed/mri_imaging/modules'))

# print(metrics.evaluate_segmentation([0,1], [1,1]))


print(tf.config.list_physical_devices(device_type=None))


def valid_dice(y_true, y_pred):
    
    dcice_value  = binary_metrics(y_true.numpy(), y_pred.numpy(), threshold_1=0.5)[0]
    return dcice_value


# dcnn_model_dir = "/home/saeed/mri_dataset_creator/"
dcnn_be_model_file = saved_models_dir + "/model_brain_exraction_dcnn_128x128.hdf5"



dcnn_be_model = tf.keras.models.load_model(dcnn_be_model_file,
                                        custom_objects={'Functional':tf.keras.models.Model, "valid_dice": valid_dice})

import os
import json
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import nibabel as nib
import numpy as np

from model import build_classifier
from deep_taylor_decomposition import add_deep_taylor_decomposition_to_model_output
from losses import relevance_guided
from metrics import sum_relevance_inner_mask, sum_relevance
from fill_model import fill_model_from_hdf5

# Set seed for experiment reproducibility
# https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
seed = 90
os.environ['PYTHONHASHSEED'] = str(seed)
tf.keras.utils.set_random_seed(seed)
tf.config.experimental.enable_op_determinism()

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../02_models_training/00_data')
target_base_path = '/mnt/neuro/nas2/Work/Graz_plus_R2star/03_results/02_performance/'

with open(os.path.join(this_file_path, '../../01_data/05_bootstrap/ASPSF_bootstrap.json'), 'r') as infile:
  ASPSF_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../../01_data/05_bootstrap/ProDem_bootstrap.json'), 'r') as infile:
  ProDem_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../01_weights_selection/trainings_filtered.json'), 'r') as infile:
  selected_weights = json.load(infile)

data_intensity_normalization_constant = 40.

for config in [
  # model                                         image                                 mask                                                  bet     rg        shape
  # ('R2star',                                      'R2star.nii.gz',                      'T1_1mm__brain_mask_@_R2star_dof6_ero5.nii.gz',       False,  False,    (208, 256, 64)),
  # ('R2star_BET',                                  'R2star.nii.gz',                      'T1_1mm__brain_mask_@_R2star_dof6_ero5.nii.gz',       True,   False,    (208, 256, 64)),
  ('R2star_RG',                                   'R2star.nii.gz',                      'T1_1mm__brain_mask_@_R2star_dof6_ero5.nii.gz',       False,  True,     (208, 256, 64)),

  # ('R2star@MNI152_dof6',                          'R2star_@_MNI152_1mm_dof6.nii.gz',    'T1_1mm__brain_mask_@_MNI152_1mm_dof6_ero5.nii.gz',   False,  False,    (182, 218, 182)),
  # ('R2star@MNI152_dof6_BET',                      'R2star_@_MNI152_1mm_dof6.nii.gz',    'T1_1mm__brain_mask_@_MNI152_1mm_dof6_ero5.nii.gz',   True,   False,    (182, 218, 182)),
  # ('R2star@MNI152_dof6_RG',                       'R2star_@_MNI152_1mm_dof6.nii.gz',    'T1_1mm__brain_mask_@_MNI152_1mm_dof6_ero5.nii.gz',   False,  True,     (182, 218, 182)),

  # ('R2star@MNI152_nlin_with_T1_mask',             'R2star_@_MNI152_1mm_nlin.nii.gz',    'T1_1mm__brain_mask_@_MNI152_1mm_nlin_ero5.nii.gz',   False,  False,    (182, 218, 182)),
  # ('R2star@MNI152_nlin_BET_with_T1_mask',         'R2star_@_MNI152_1mm_nlin.nii.gz',    'T1_1mm__brain_mask_@_MNI152_1mm_nlin_ero5.nii.gz',   True,   False,    (182, 218, 182)),
  # ('R2star@MNI152_nlin_RG',                       'R2star_@_MNI152_1mm_nlin.nii.gz',    'T1_1mm__brain_mask_@_MNI152_1mm_nlin_ero5.nii.gz',   False,  True,     (182, 218, 182)),
  
  # ('R2star@MNI152_nlin_with_MNI_mask',            'R2star_@_MNI152_1mm_nlin.nii.gz',    'MNI152_T1_1mm_brain_mask_dil_ero7.nii.gz',           True,   False,    (182, 218, 182)),
  # ('R2star@MNI152_nlin_RG_MNI',                   'R2star_@_MNI152_1mm_nlin.nii.gz',    'MNI152_T1_1mm_brain_mask_dil_ero7.nii.gz',           False,  True,     (182, 218, 182)),
]:
  (model_name, input_image, input_mask, use_bet, use_rg, input_shape) = config
  
  for selected_weight in selected_weights[model_name]:
    if len(selected_weight['selected']) == 0:
      continue

    # selection is done with visible device...
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # ... and not tf.device because it fails when first card is already in use
    with tf.device('/device:GPU:0'):
      K.clear_session()
  
      # classifier
      (input_tensor, output_tensor) = build_classifier(input_shape + (1,), filters=8, kernel_initializer='he_uniform') 
        
      # relevance-guided extension
      model = Model(inputs=input_tensor, outputs=output_tensor)

      fill_model_from_hdf5(
        model,
        os.path.join(this_file_path, '../../02_models_training/03_training/weights', 'R2star_RG', 'R2star__boostrap_index-05__initial_weights_index-02__060-tcl-0.329-vcl-0.338-tca-0.855-vca-0.896-tsrim-0.982-vsrim-0.985.h5'),
        custom_objects={
          'relevance_guided': relevance_guided,
          'sum_relevance_inner_mask': sum_relevance_inner_mask,
          'sum_relevance': sum_relevance,
        },
        compile=True,
      )
      model.summary()
      exit()

      model = Model(inputs=model.input, outputs=model.get_layer('3dconv_encoding_pooling_1').output)
      
      for scan in (ASPSF_bootstrap[5]['test'] + ProDem_bootstrap[5]['test']):
        # image
        image = nib.load(os.path.join(data_base_path, scan, input_image))
        image_data = image.get_fdata(caching='unchanged')
        # image_data = image_data[..., 1]
        image_data = np.nan_to_num(image_data, copy=False, posinf=0., neginf=0.)

        # normalize image
        image_data /= data_intensity_normalization_constant

        # mask
        mask = nib.load(os.path.join(data_base_path, scan, input_mask))
        mask_data = mask.get_fdata(caching='unchanged')
        mask_data = np.nan_to_num(mask_data, copy=False, posinf=0., neginf=0.)

        if use_bet:
          image_data *= mask_data

        prediction = model.predict(image_data[None, :, :, :, None])

        print(scan)
        # print(prediction)
        print(prediction.shape)
        np.savetxt('tf_test.txt', prediction[0, :, :, :, 0].flatten())
        # prediction.savetxt()
        exit()

        # print(scan + ' ' + str(prediction[0][0][0]) + ' ' + str(prediction[0][0][1]))
      
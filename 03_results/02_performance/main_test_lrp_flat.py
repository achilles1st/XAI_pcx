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
  ('R2star_BET',                                  'R2star.nii.gz',                      'T1_1mm__brain_mask_@_R2star_dof6_ero5.nii.gz',       True,   False,    (208, 256, 64)),
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
    # selection is done with visible device...
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # ... and not tf.device because it fails when first card is already in use
    with tf.device('/device:GPU:0'):
      K.clear_session()
  
      # classifier
      (input_tensor, output_tensor) = build_classifier(input_shape + (1,), filters=8, kernel_initializer='he_uniform') 
        
      # relevance-guided extension
      heatmap_tensor = add_deep_taylor_decomposition_to_model_output(output_tensor)
      model = Model(inputs=input_tensor, outputs=[output_tensor, heatmap_tensor])

      fill_model_from_hdf5(
        model,
        os.path.join(this_file_path, '../../02_models_training/03_training/weights', model_name, selected_weight['file']),
        custom_objects={
          'relevance_guided': relevance_guided,
          'sum_relevance_inner_mask': sum_relevance_inner_mask,
          'sum_relevance': sum_relevance,
        },
        compile=True,
      )
      
      heatmaps_path = os.path.join(target_base_path, 'heatmaps_lrp_flat', model_name)
      if not os.path.exists(heatmaps_path):
        os.mkdir(heatmaps_path)

      final_heatmaps_path = heatmaps_path + '/boostrap_index-{0:02d}__initial_weights_index-{1:02d}'.format(selected_weight['bootstrap_index'], selected_weight['initial_weights_index'])
      if not os.path.exists(final_heatmaps_path):
        os.mkdir(final_heatmaps_path)

      classifications_result = {}

      for scan in (ASPSF_bootstrap[selected_weight['bootstrap_index']]['test'] + ProDem_bootstrap[selected_weight['bootstrap_index']]['test']):
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

        classifications_result[scan] = {
          'ASPSF': prediction[0][0][0].item(),
          'ProDem': prediction[0][0][1].item(),
        }

        print(scan + ' ' + str(prediction[0][0][0]) + ' ' + str(prediction[0][0][1]))
        heatmap = nib.Nifti1Image(np.squeeze(prediction[1]), image.affine, header=image.header)
        nib.save(heatmap, os.path.join(final_heatmaps_path, '{0}_{1:.3f}_{2:.3f}.nii.gz'.format(scan, prediction[0][0][0], prediction[0][0][1])))
      
      classifications_path = os.path.join(target_base_path, 'classifications', model_name)
      if not os.path.exists(classifications_path):
        os.mkdir(classifications_path)

      with open(classifications_path + '/classification__boostrap_index-{0:02d}__initial_weights_index-{1:02d}.json'.format(selected_weight['bootstrap_index'], selected_weight['initial_weights_index']), 'w') as outfile:
        json.dump(classifications_result, outfile, indent=2)

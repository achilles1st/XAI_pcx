import sys
import os
import subprocess

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../02_models_training/00_data')
target_base_path = '/mnt/neuro/nas2/Work/Graz_plus_R2star/03_results/02_performance'

configs = [
  # model                                         warp file                               premat file                     invert
  ('R2star',                                      'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    False),
  ('R2star_BET',                                  'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    False),
  ('R2star_RG',                                   'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    False),

  # ('R2star@MNI152_dof6',                          'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    True),
  # ('R2star@MNI152_dof6_BET',                      'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    True),
  # ('R2star@MNI152_dof6_RG',                       'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    True),

  # ('R2star@MNI152_nlin_with_T1_mask',             None,                                   None,                           False),
  # ('R2star@MNI152_nlin_BET_with_T1_mask',         None,                                   None,                           False),
  # ('R2star@MNI152_nlin_RG',                       None,                                   None,                           False),
  
  # ('R2star@MNI152_nlin_with_MNI_mask',            None,                                   None,                           False),
  # ('R2star@MNI152_nlin_RG_MNI',                   None,                                   None,                           False),
]

def run(config):
  (model_name, warp_file, premat_file, invert) = config

  registered_heatmaps_abs_path = os.path.join(target_base_path, 'heatmaps_warped', model_name)

  # sum
  params = [
    'fslmaths'
  ]
  isfirst = True
  count = 0

  for run_name in os.listdir(registered_heatmaps_abs_path):
    if not isfirst:
      params.append('-add')

    params.append(os.path.join(registered_heatmaps_abs_path, run_name, 'sum.nii.gz'))
    isfirst = False

  params.append(os.path.join(registered_heatmaps_abs_path, 'sum.nii.gz'))

  print(' '.join(params))
  subprocess.call(params)

print(sys.argv)
index = int(sys.argv[1])
print(index)

run(configs[index])

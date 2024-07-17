import sys
import os
import subprocess

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../02_models_training/00_data')
target_base_path = '/mnt/neuro/nas2/Work/Graz_plus_R2star/03_results/02_performance'
heatmaps_base_path = os.path.join(target_base_path, 'heatmaps')

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
  if not os.path.exists(registered_heatmaps_abs_path):
    os.mkdir(registered_heatmaps_abs_path)

  heatmaps_abs_path = os.path.join(heatmaps_base_path, model_name)
  for run_name in os.listdir(heatmaps_abs_path):
    if not os.path.exists(os.path.join(registered_heatmaps_abs_path, run_name)):
      os.mkdir(os.path.join(registered_heatmaps_abs_path, run_name))

    for file_name in os.listdir(os.path.join(heatmaps_abs_path, run_name)):
      if file_name.endswith('.nii.gz'):
        subject_name = file_name[0:16]

      if invert:
        inverted_premat_file = os.path.join(registered_heatmaps_abs_path, run_name, subject_name + '_premat_inv.mat')
        invert_args = [
          'convert_xfm',
          '-omat',
          inverted_premat_file,
          '-inverse',
          os.path.join(data_base_path, subject_name, premat_file)
        ]

        print(' '.join(invert_args))
        subprocess.call(invert_args)
      else:
        inverted_premat_file = None

      if warp_file is not None:
        warp_args = [
          'applywarp',
          '--in=' + os.path.join(heatmaps_abs_path, run_name, file_name),
          '--out=' + os.path.join(registered_heatmaps_abs_path, run_name, file_name),
          '--ref=/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz',
          '--warp=' + os.path.join(data_base_path, subject_name, warp_file)
        ]
        if inverted_premat_file is not None:
          warp_args.append('--premat=' + inverted_premat_file)
        elif premat_file is not None:
          warp_args.append('--premat=' + os.path.join(data_base_path, subject_name, premat_file))

        print(' '.join(warp_args))
        subprocess.call(warp_args)
      else:
        cp_args = [
          'cp',
          os.path.join(heatmaps_abs_path, run_name, file_name),
          os.path.join(registered_heatmaps_abs_path, run_name, file_name),
        ]

        print(' '.join(cp_args))
        subprocess.call(cp_args)

print(sys.argv)
index = int(sys.argv[1])
print(index)

run(configs[index])
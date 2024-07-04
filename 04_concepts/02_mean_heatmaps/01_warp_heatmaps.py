import sys
import os
import subprocess

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../02_models_training/00_data')
target_base_path = '/mnt/neuro/nas2/Work/Graz_plus_R2star/04_concepts/01_heatmaps/'
# /mnt/neuro/nas2/Work/Graz_plus_R2star/04_concepts/01_heatmaps/deep_taylor_decomposition/relevance/ref_images/R2star/Both

configs = [
  # model             warp file                               premat file                     invert    class
  ('R2star_RG',			  'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    False,    'AD'),
  ('R2star_RG',			  'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    False,    'NC'),
  ('R2star_RG',			  'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    False,    'Both'),
]

def run(config):
  (model_name, warp_file, premat_file, invert, class_name) = config

  registered_heatmaps_abs_path = os.path.join(target_base_path, 'deep_taylor_decomposition', 'relevance', 'ref_images', model_name, f'{class_name}_warped')
  if not os.path.exists(registered_heatmaps_abs_path):
    os.makedirs(registered_heatmaps_abs_path)

  heatmaps_abs_path = os.path.join(target_base_path, 'deep_taylor_decomposition', 'relevance', 'ref_images', model_name, class_name)

  for file_name in os.listdir(os.path.join(heatmaps_abs_path)):
    if file_name.endswith('.nii.gz'):
      # rel_0.351__channel_01__data_index_73__053401__20130301__0.149_0.851.nii.gz
      start_of_subject_name = len('rel_0.351__channel_01__data_index_73__')
      subject_name = file_name[start_of_subject_name:start_of_subject_name + 16]

    if invert:
      inverted_premat_file = os.path.join(registered_heatmaps_abs_path, subject_name + '_premat_inv.mat')
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
        '--in=' + os.path.join(heatmaps_abs_path, file_name),
        '--out=' + os.path.join(registered_heatmaps_abs_path, file_name),
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
        os.path.join(heatmaps_abs_path, file_name),
        os.path.join(registered_heatmaps_abs_path, file_name),
      ]

      print(' '.join(cp_args))
      subprocess.call(cp_args)

print(sys.argv)
index = int(sys.argv[1])
print(index)

run(configs[index])
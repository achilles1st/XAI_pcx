import os
import itertools
import subprocess

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../02_models_training/00_data')
target_base_path = '/mnt/neuro/nas2/Work/Graz_plus_R2star/04_concepts/01_heatmaps/'

mean_count = 10

for model_name, _, _, _ in [
  # model             warp file                               premat file                     invert
  # ('R2star_RG_BG',    'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    False),
  ('R2star_RG',			  'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    False),
  # ('R2star_BET',		  'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    False),
  # ('R2star',				  'T1_1mm_to_MNI152_warpcoeff.nii.gz',    'R2star_to_T1_1mm_dof6.mat',    False),
]:
  for class_name in [
    # 'AD',
    # 'NC',
    'Both',
  ]:
    mean_registered_heatmaps_abs_path = os.path.join(target_base_path, 'deep_taylor_decomposition', 'relevance', 'ref_images', model_name, f'{class_name}_warped_mean_{str(mean_count)}')
    if not os.path.exists(mean_registered_heatmaps_abs_path):
      os.makedirs(mean_registered_heatmaps_abs_path)

    registered_heatmaps_abs_path = os.path.join(target_base_path, 'deep_taylor_decomposition', 'relevance', 'ref_images', model_name, f'{class_name}_warped')
    heatmap_files = os.listdir(os.path.join(registered_heatmaps_abs_path))
    rel_len = len('rel_0.351')

    for channel_index, channel_files in itertools.groupby(heatmap_files, lambda x: int(x[len('rel_0.351__channel_'):len('rel_0.351__channel_') + 2])):
      channel_files = list(channel_files)
      channel_files.sort()

      # top
      params = [
        'fslmaths'
      ]
      isfirst = True
      for data_index in range(mean_count):
        if not isfirst:
          params.append('-add')
        params.append(os.path.join(registered_heatmaps_abs_path, channel_files[data_index]))
        isfirst = False
      
      params.append(os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__top.nii.gz'))
      print(' '.join(params))
      subprocess.call(params)

      # bottom
      start_index = len(channel_files) - mean_count
      params = [
        'fslmaths'
      ]
      isfirst = True
      for data_index in range(mean_count):
        if not isfirst:
          params.append('-add')
        params.append(os.path.join(registered_heatmaps_abs_path, channel_files[start_index + data_index]))
        isfirst = False
      
      params.append(os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__bottom.nii.gz'))
      print(' '.join(params))
      subprocess.call(params)

      # middle
      start_index = (len(channel_files) - mean_count) // 2
      params = [
        'fslmaths'
      ]
      isfirst = True
      for data_index in range(mean_count):
        if not isfirst:
          params.append('-add')
        params.append(os.path.join(registered_heatmaps_abs_path, channel_files[start_index + data_index]))
        isfirst = False
      
      params.append(os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__middle.nii.gz'))
      print(' '.join(params))
      subprocess.call(params)

      # difference top-bottom
      params = [
        'fslmaths',
        os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__top.nii.gz'),
        '-sub',
        os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__bottom.nii.gz'),
        os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__difference_top_bottom.nii.gz'),
      ]
      print(' '.join(params))
      subprocess.call(params)

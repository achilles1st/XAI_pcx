import os
import itertools
import subprocess

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../02_models_training/00_data')
target_base_path = '/mnt/neuro/nas2/Work/Graz_plus_R2star/04_concepts/01_heatmaps/'

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
    mean_registered_heatmaps_abs_path = os.path.join(target_base_path, 'deep_taylor_decomposition', 'relevance', 'ref_images', model_name, f'{class_name}_warped_class_mean')
    if not os.path.exists(mean_registered_heatmaps_abs_path):
      os.makedirs(mean_registered_heatmaps_abs_path)

    registered_heatmaps_abs_path = os.path.join(target_base_path, 'deep_taylor_decomposition', 'relevance', 'ref_images', model_name, f'{class_name}_warped')
    heatmap_files = os.listdir(os.path.join(registered_heatmaps_abs_path))
    rel_len = len('rel_0.351')

    for channel_index, channel_files in itertools.groupby(heatmap_files, lambda x: int(x[len('rel_0.351__channel_'):len('rel_0.351__channel_') + 2])):
      channel_files = list(channel_files)
      channel_files.sort()

      # CN
      params = [
        'fslmaths'
      ]
      count = 0
      isfirst = True
      for channel_file in channel_files:
        # rel_0.415__channel_00__data_index_73__205100__20141111__0.047_0.953.nii.gz
        subject_id = channel_file[len('rel_0.415__channel_00__data_index_73__'):len('rel_0.415__channel_00__data_index_73__205100__20141111')]
        if subject_id.startswith('8'):
          continue

        if not isfirst:
          params.append('-add')
        params.append(os.path.join(registered_heatmaps_abs_path, channel_file))
        isfirst = False
        count = count + 1
      
      params.append(os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__CN_sum.nii.gz'))
      print(' '.join(params))
      subprocess.call(params)

      params = [
        'fslmaths',
        os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__CN_sum.nii.gz'),
        '-div',
        str(count),
        os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__CN_mean.nii.gz'),
      ]
      print(' '.join(params))
      subprocess.call(params)

      # AD
      params = [
        'fslmaths'
      ]
      count = 0
      isfirst = True
      for channel_file in channel_files:
        # rel_0.415__channel_00__data_index_73__205100__20141111__0.047_0.953.nii.gz
        subject_id = channel_file[len('rel_0.415__channel_00__data_index_73__'):len('rel_0.415__channel_00__data_index_73__205100__20141111')]
        if not subject_id.startswith('8'):
          continue

        if not isfirst:
          params.append('-add')
        params.append(os.path.join(registered_heatmaps_abs_path, channel_file))
        isfirst = False
        count = count + 1
      
      params.append(os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__AD_sum.nii.gz'))
      print(' '.join(params))
      subprocess.call(params)

      params = [
        'fslmaths',
        os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__AD_sum.nii.gz'),
        '-div',
        str(count),
        os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__AD_mean.nii.gz'),
      ]
      print(' '.join(params))
      subprocess.call(params)

      # difference CN-AD
      params = [
        'fslmaths',
        os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__CN_mean.nii.gz'),
        '-sub',
        os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__AD_mean.nii.gz'),
        os.path.join(mean_registered_heatmaps_abs_path, f'{channel_files[0][0:rel_len]}__channel_{channel_index:02d}__CN_AD_mean_diff.nii.gz'),
      ]
      print(' '.join(params))
      subprocess.call(params)

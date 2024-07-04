import sys
import os
import subprocess

run_name = '/mnt/neuro/nas2/Work/Graz_plus_R2star/03_results/02_performance/heatmaps_warped/R2star_RG/boostrap_index-05__initial_weights_index-02'

# CN
params = [
  'fslmaths'
]
count = 0
isfirst = True
for image_file in os.listdir(run_name):
  if image_file.endswith('.nii.gz') and image_file not in ['sum.nii.gz', 'mean.nii.gz', 'AD_sum.nii.gz', 'CN_sum.nii.gz', 'AD_mean.nii.gz', 'CN_mean.nii.gz', 'CN_AD_mean_diff.nii.gz']:
    if image_file.startswith('8'):
      continue

    if not isfirst:
      params.append('-add')
    params.append(os.path.join(run_name, image_file))
    isfirst = False
    count = count + 1
      
params.append(os.path.join(run_name, 'CN_sum.nii.gz'))
print(' '.join(params))
subprocess.call(params)

params = [
  'fslmaths',
  os.path.join(run_name, 'CN_sum.nii.gz'),
  '-div',
  str(count),
  os.path.join(run_name, 'CN_mean.nii.gz'),
]
print(' '.join(params))
subprocess.call(params)

# AD
params = [
  'fslmaths'
]
count = 0
isfirst = True
for image_file in os.listdir(run_name):
  if image_file.endswith('.nii.gz') and image_file not in ['sum.nii.gz', 'mean.nii.gz', 'AD_sum.nii.gz', 'CN_sum.nii.gz', 'AD_mean.nii.gz', 'CN_mean.nii.gz', 'CN_AD_mean_diff.nii.gz']:
    if not image_file.startswith('8'):
      continue

    if not isfirst:
      params.append('-add')
    params.append(os.path.join(run_name, image_file))
    isfirst = False
    count = count + 1
      
params.append(os.path.join(run_name, 'AD_sum.nii.gz'))
print(' '.join(params))
subprocess.call(params)

params = [
  'fslmaths',
  os.path.join(run_name, 'AD_sum.nii.gz'),
  '-div',
  str(count),
  os.path.join(run_name, 'AD_mean.nii.gz'),
]
print(' '.join(params))
subprocess.call(params)

# difference CN-AD
params = [
  'fslmaths',
  os.path.join(run_name, 'CN_mean.nii.gz'),
  '-sub',
  os.path.join(run_name, 'AD_mean.nii.gz'),
  os.path.join(run_name, 'CN_AD_mean_diff.nii.gz'),
]
print(' '.join(params))
subprocess.call(params)
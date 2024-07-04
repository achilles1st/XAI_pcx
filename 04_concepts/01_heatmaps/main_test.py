import os
import json
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn

from crp.concepts import ChannelConcept
from crp.helper import get_layer_names, load_maximization
from crp.attribution import CondAttribution

from zennit.composites import EpsilonPlus, EpsilonAlpha2Beta1

from composites import DeepTaylorDecompositionComposite
from model import Model
from dataset import NiftiDataset
from h5loader import load
from visualization import FeatureVisualizationTarget

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../02_models_training/00_data')
target_base_path = '/mnt/neuro/nas2/Work/Graz_plus_R2star/04_concepts/01_heatmaps/'

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False

with open(os.path.join(this_file_path, '../../01_data/05_bootstrap/ASPSF_bootstrap.json'), 'r') as infile:
  ASPSF_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../../01_data/05_bootstrap/ProDem_bootstrap.json'), 'r') as infile:
  ProDem_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../../03_results/01_weights_selection/trainings_highest.json'), 'r') as infile:
  selected_weights = json.load(infile)

data_intensity_normalization_constant = 40.

# device
device = (
  'cuda:0'
  if torch.cuda.is_available()
  else 'cpu'
)
print(f'Using {device} device')

for config in [
  #name						    image								mask                                              bet
#   ('R2star_RG_BG',		'R2star.nii.gz',		'T1_1mm__brain_mask_@_R2star_dof6_ero5.nii.gz',		False),
  ('R2star_RG',				'R2star.nii.gz',		'T1_1mm__brain_mask_@_R2star_dof6_ero5.nii.gz',		False),
#   ('R2star_BET',			'R2star.nii.gz',		'T1_1mm__brain_mask_@_R2star_dof6_ero5.nii.gz',		True),
#   ('R2star',					'R2star.nii.gz',		'T1_1mm__brain_mask_@_R2star_dof6_ero5.nii.gz',		False),
]:
	name, input_image, mask, bet = config
	
  # target
	training_target_base_path = target_base_path # os.path.join(target_base_path, name)
	if not os.path.exists(training_target_base_path):
		os.makedirs(training_target_base_path)

	training_data = selected_weights[name]
	bootstrap_index = training_data['bootstrap_index']
	weight_file = training_data['file']

	# print(training_data)

	# dataset
	data_base_path = os.path.join(this_file_path, '../../02_models_training/00_data')

	with open(os.path.join(this_file_path, '../../01_data/05_bootstrap/ASPSF_bootstrap.json'), 'r') as infile:
		ASPSF_bootstrap = json.load(infile)

	with open(os.path.join(this_file_path, '../../01_data/05_bootstrap/ProDem_bootstrap.json'), 'r') as infile:
		ProDem_bootstrap = json.load(infile)

	# model
	# print(name)
	# print(weight_file)
	model = Model(with_softmax=True).to(device)
	model.load_state_dict(load(os.path.join('../../02_models_training/03_training/weights', name, weight_file)))

	# ones = torch.ones((1, 1, 208, 256, 64)).float().to(device)

	# image_data_tensor = torch.from_numpy(image_data[None, None, :, :, :]).float().to(device)
	# ones = torch.ones((1, 1, 208, 256, 64)).float().to(device)
	# result = model(ones)
	# print(result)

	for scan in (ASPSF_bootstrap[bootstrap_index]['test'] + ProDem_bootstrap[bootstrap_index]['test']):
		# image
		image = nib.load(os.path.join(data_base_path, scan, input_image))
		image_data = image.get_fdata(caching='unchanged')
        # image_data = image_data[..., 1]
		image_data = np.nan_to_num(image_data, copy=False, posinf=0., neginf=0.)

        # normalize image
		image_data = image_data / data_intensity_normalization_constant

		image_data_tensor = torch.from_numpy(image_data[None, None, :, :, :]).float().to(device)
		result = model(image_data_tensor)
		print(scan)
		print(result)
		# print(result.shape)
		# np.savetxt('pt_test.txt', result[0, 0, :, :, :].cpu().detach().numpy().flatten())
		# exit()

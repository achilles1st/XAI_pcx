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
target_base_path = 'C:/Users/tosic/StefanTosic/04_concepts/01_heatmaps/'

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
	name, image, mask, bet = config
	
  # target
	training_target_base_path = target_base_path # os.path.join(target_base_path, name)
	if not os.path.exists(training_target_base_path):
		os.makedirs(training_target_base_path)

	training_data = selected_weights[name]
	bootstrap_index = training_data['bootstrap_index']
	weight_file = training_data['file']

	# dataset
	data_base_path = os.path.normpath(os.path.join(this_file_path, '../../02_models_training/00_data'))

	with open(os.path.join(this_file_path, '../../01_data/05_bootstrap/ASPSF_bootstrap.json'), 'r') as infile:
		ASPSF_bootstrap = json.load(infile)

	with open(os.path.join(this_file_path, '../../01_data/05_bootstrap/ProDem_bootstrap.json'), 'r') as infile:
		ProDem_bootstrap = json.load(infile)

	test_data_paths = [os.path.join(data_base_path, subject_folder) for subject_folder in ASPSF_bootstrap[bootstrap_index]['test'] + ProDem_bootstrap[bootstrap_index]['test']]
	test_set = NiftiDataset(
		device,
		test_data_paths,
		[
			image
		],
		[
			mask
		],
		(208, 256, 64),
		data_intensity_normalization_constant,
		use_mask_on_input=bet,
		output_only_categories=True,
	)

	for composite_name, composite in [
		('deep_taylor_decomposition', DeepTaylorDecompositionComposite()), # define DTD rules and canonizers in zennit
		# ('epsilon_plus', EpsilonPlus()),
		# ('epsilon_alpha2_beta1', EpsilonAlpha2Beta1())
	]:
		composite_target_base_path = os.path.join(training_target_base_path, composite_name)
		if not os.path.exists(composite_target_base_path):
			os.mkdir(composite_target_base_path)

		for mode, mode_targets, use_start_layer in [
			('relevance', [('NC', [0]), ('AD', [1]), ('Both', [0, 1])], False),
			# ('activation', [('all', None)], True),
		]:
			mode_target_base_path = os.path.join(composite_target_base_path, mode)

			if not os.path.exists(os.path.join(mode_target_base_path, 'ref_images', name)):
				os.makedirs(os.path.join(mode_target_base_path, 'ref_images', name))
			
			# if not os.path.exists(os.path.join(mode_target_base_path, 'ref_images_zoomed', name)):
			# 	os.makedirs(os.path.join(mode_target_base_path, 'ref_images_zoomed', name))

			# model and weights
			model = Model(with_softmax=True).to(device)
			model.load_state_dict(load(os.path.join('../../02_models_training/03_training/weights', name, weight_file)))

			for targets_name, targets in mode_targets:
				if not os.path.exists(os.path.join(mode_target_base_path, 'ref_images', name, targets_name)):
					os.makedirs(os.path.join(mode_target_base_path, 'ref_images', name, targets_name))
			
				# if not os.path.exists(os.path.join(mode_target_base_path, 'ref_images_zoomed', name, targets_name)):
				# 	os.makedirs(os.path.join(mode_target_base_path, 'ref_images_zoomed', name, targets_name))

				# here, each channel is defined as a concept
				# or define your own notion!
				cc = ChannelConcept()

				# get layer names of Conv3D and Linear layers
				layer_names = get_layer_names(model, [nn.Conv3d, nn.Linear])
				layer_map = { layer : cc for layer in layer_names }
				# layer_map = { 'down_conv5': cc}

				# load CRP toolbox
				attribution = CondAttribution(model)
				
				# feature vis
				fv = FeatureVisualizationTarget(
					attribution, test_set, layer_map, preprocess_fn=None, abs_norm=False,
					path=os.path.join(mode_target_base_path, 'stats', name, targets_name),
					targets=targets, use_start_layer=use_start_layer)
				fv.run(composite, 0, len(test_set), 6, 1, device)

				concept_ids = [0, 1, 2, 3, 4, 5, 6, 7]
				r_range = (0, len(test_set))

				vals_per_class_and_concept = np.zeros((len(concept_ids), 2))
				for c_id in concept_ids:
					t, rel = fv.compute_stats(c_id, 'down_conv5', mode, top_N=2, mean_N=len(test_set), norm=False)
					vals_per_class_and_concept[c_id, t] = rel.numpy()
					# print(t)
					# print(rel)

				concept_stats_path = os.path.join(mode_target_base_path, 'down_conv5_stats', name, targets_name)
				if not os.path.exists(concept_stats_path):
					os.makedirs(concept_stats_path)
				with open(os.path.join(concept_stats_path, 'vals_per_concept_and_class.json'), 'w') as outfile:
					json.dump(vals_per_class_and_concept.tolist(), outfile)

				# sum vals per concept
				print(vals_per_class_and_concept)
				vals_per_concept = (vals_per_class_and_concept / vals_per_class_and_concept.sum()).sum(axis=1)
				print(vals_per_concept)

				ref_c = fv.get_max_reference(concept_ids, 'down_conv5', mode, r_range, composite=composite, plot_fn=None, batch_size=6)
				d_c_sorted, rel_c_sorted, rf_c_sorted = load_maximization(os.path.join(mode_target_base_path, 'stats', name, targets_name, 'RelMax_sum_unnormed' if mode == 'relevance' else 'ActMax_sum_unnormed'), 'down_conv5')
				# print(d_c_sorted)
				# print(rel_c_sorted)
				# print(rf_c_sorted)
				for c_id in concept_ids:
					d_indices = d_c_sorted[r_range[0]:r_range[1], c_id]
					# print(rel_c_sorted[r_range[0]:r_range[1], c_id])
					for d_id_index, d_id in enumerate(d_indices):
						data_path = test_set.data_paths[d_id]
						subject_name = data_path.split('\\')[-1]
						image = nib.load(os.path.join(data_path, 'R2star.nii.gz'))
						image_data, _ = test_set[d_id]
						image_data = torch.unsqueeze(image_data, 0)
						result = model(image_data.to(device))
						heatmap = nib.Nifti1Image(ref_c[c_id][1][d_id_index], image.affine, header=image.header)
						nib.save(heatmap, os.path.normpath(os.path.join(mode_target_base_path, 'ref_images', name, targets_name, f'rel_{vals_per_concept[c_id]:.3f}__channel_{c_id:02d}__data_index_{d_id_index:02d}__{os.path.basename(subject_name)}__{result[0][0]:.3f}_{result[0][1]:.3f}.nii.gz')))

				# ref_c = fv.get_max_reference(concept_ids, 'down_conv5', mode, r_range, composite=composite, rf=True, plot_fn=None, batch_size=6)
				# d_c_sorted, rel_c_sorted, rf_c_sorted = load_maximization(os.path.join(mode_target_base_path, 'stats', name, targets_name, 'RelMax_sum_unnormed' if mode == 'relevance' else 'ActMax_sum_unnormed'), 'down_conv5')
				# for c_id in concept_ids:
				# 	d_indices = d_c_sorted[r_range[0]:r_range[1], c_id]
				# 	print(rel_c_sorted[r_range[0]:r_range[1], c_id])
				# 	for d_id_index, d_id in enumerate(d_indices):
				# 		data_path = test_set.data_paths[d_id]
				# 		subject_name = data_path.split('/')[-1]
				# 		image = nib.load(os.path.join(data_path, 'R2star.nii.gz'))
				# 		image_data, _ = test_set[d_id_index]
				# 		image_data = torch.unsqueeze(image_data, 0)
				# 		result = model(image_data.to(device))
				# 		heatmap = nib.Nifti1Image(ref_c[c_id][1][d_id_index], image.affine, header=image.header)
				# 		nib.save(heatmap, os.path.join(mode_target_base_path, 'ref_images_zoomed', name, targets_name, f'rel_{vals_per_concept[c_id]:.3f}__channel_{c_id:02d}__data_index_{d_id_index:02d}__{subject_name}__{result[0][0]:.3f}_{result[0][1]:.3f}.nii.gz'))

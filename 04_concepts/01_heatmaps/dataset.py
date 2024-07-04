import os
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset

class NiftiDataset(Dataset):
  def __init__(self, device, 
               data_paths, input_names, mask_names,
               input_shape, normalization_constant,
               use_mask_on_input=False,
               output_only_categories=False):
    self.device = device
    self.data_paths = data_paths
    self.input_names = input_names
    self.mask_names = mask_names
    self.input_shape = input_shape
    self.normalization_constant = normalization_constant
    self.use_mask_on_input = use_mask_on_input
    self.output_only_categories = output_only_categories
    self.filepaths_with_category_and_sample_weighting = []

    for data_path in self.data_paths:
      subject_dirname = data_path.split('/')[-1]
      
      category_index = 1 if subject_dirname.startswith('8') else 0

      self.filepaths_with_category_and_sample_weighting.append((
        [os.path.join(data_path, input_name) for input_name in self.input_names],
        [os.path.join(data_path, mask_name) for mask_name in self.mask_names],
        category_index,
        1., # sample_weighting
      ))

  def __len__(self):
    return len(self.filepaths_with_category_and_sample_weighting)

  def __getitem__(self, index):
    filepaths_with_category = self.filepaths_with_category_and_sample_weighting[index]
    input_paths, mask_paths, category_index, sample_weighting = filepaths_with_category
    
    inputs = []
    masks = []
    for input_path_index in range(0, len(input_paths)):
      input_path = input_paths[input_path_index]
      mask_path = mask_paths[input_path_index]

      # image
      image = nib.load(input_path)
      # if image.shape != self.input_shape:
      #   print(input_path)
      #   print(image.shape)
        
      image_data = image.get_fdata(caching='unchanged')
      # image_data = image_data[..., 1]
      image_data = np.nan_to_num(image_data, copy=False, nan=0., posinf=0., neginf=0.)
      # if np.isnan(np.sum(image_data)):
      #   print(input_path + ' has NaNs.')
      # if np.max(image_data) == 0.:
      #   print(input_path + ' max is 0.')

      # mask
      mask = nib.load(mask_path)
      # if mask.shape != self.input_shape:
      #   print(mask_path)
      #   print(mask.shape)
        
      mask_data = mask.get_fdata(caching='unchanged')
      # if np.isnan(np.sum(mask_data)):
      #   print(mask_path + ' has NaNs.')
      # if np.max(mask_data) == 0.:
      #   print(mask_path + ' max is 0.')

      if self.use_mask_on_input:
        image_data *= mask_data

      if self.normalization_constant != None:
        image_data = image_data / self.normalization_constant

      inputs.append(image_data)
      masks.append(mask_data)
    
    inputs = np.array(inputs)
    masks = np.array(masks)

    if self.output_only_categories:
      return torch.tensor(inputs, dtype=torch.float), category_index
    return torch.tensor(inputs, dtype=torch.float), torch.tensor(masks, dtype=torch.float), category_index
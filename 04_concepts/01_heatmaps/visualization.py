from typing import Dict, Callable
import math

import torch

from crp.visualization import FeatureVisualization
from crp.attribution import CondAttribution
from crp.concepts import Concept, ChannelConcept
from crp.cache import Cache

class FeatureVisualizationTarget(FeatureVisualization):
	def __init__(self, attribution: CondAttribution, dataset, layer_map: Dict[str, Concept], preprocess_fn: Callable=None,
							 max_target='sum', abs_norm=True, path='FeatureVisualization', device=None, cache: Cache=None,
							 targets=None, use_start_layer=True):
		super().__init__(attribution, dataset, layer_map, preprocess_fn, max_target, abs_norm, path, device, cache)
		self.targets = targets
		self.use_start_layer = use_start_layer

		# in the base class SAMPLE_SIZE is fixed to 40
		self.RelMax.SAMPLE_SIZE = len(dataset)
		self.ActMax.SAMPLE_SIZE = len(dataset)
		self.RelStats.SAMPLE_SIZE = len(dataset)
		self.ActStats.SAMPLE_SIZE = len(dataset)
	
	def _attribution_on_reference(self, data, concept_id: int, layer_name: str, composite, rf=False, neuron_ids: list=[], batch_size=32):
		n_samples = len(data)
		if n_samples > batch_size:
			batches = math.ceil(n_samples / batch_size)
		else:
			batches = 1
			batch_size = n_samples

		if rf and (len(neuron_ids) != n_samples):
			raise ValueError('length of "neuron_ids" must be equal to the length of "data"')

		heatmaps = []
		for b in range(batches):
			data_batch = data[b * batch_size: (b + 1) * batch_size].detach().requires_grad_()

			if rf:
				neuron_ids = neuron_ids[b * batch_size: (b + 1) * batch_size]
				conditions = [{layer_name: {concept_id: n_index}} | ({'y': self.targets} if self.targets is not None else {}) for n_index in neuron_ids]
				attr = self.attribution(data_batch, conditions, composite, mask_map=ChannelConcept.mask_rf, start_layer=layer_name if self.use_start_layer else None, on_device=self.device, 
																exclude_parallel=False)
			else:
				conditions = [{layer_name: [concept_id]} | ({'y': self.targets} if self.targets is not None else {})]
				attr = self.attribution(data_batch, conditions, composite, start_layer=layer_name if self.use_start_layer else None, on_device=self.device, exclude_parallel=False)

			heatmaps.append(attr.heatmap)
		
		return torch.cat(heatmaps, dim=0)
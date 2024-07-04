import torch
from zennit.types import Activation, Convolution
from zennit.composites import register_composite, LayerMapComposite
from zennit.rules import Pass

from rules import DeepTaylorDecompositionHook

@register_composite('deep_taylor_decomposition')
class DeepTaylorDecompositionComposite(LayerMapComposite):
  def __init__(self, canonizers=None):
    layer_map = [
      (torch.nn.Softmax, Pass()),
      (Activation, Pass()),
      (Convolution, DeepTaylorDecompositionHook()),
      (torch.nn.Linear, DeepTaylorDecompositionHook()),
    ]
    super().__init__(layer_map, canonizers=canonizers)
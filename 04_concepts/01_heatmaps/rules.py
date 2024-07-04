import torch
from zennit.rules import BasicHook

class DeepTaylorDecompositionHook(BasicHook):
	def __init__(self):
	  super().__init__(
			input_modifiers=[
				lambda input: input + 1e-7,
			],
			param_modifiers=[
				lambda param, name: param.clamp(min=1e-7) if name != 'bias' else torch.zeros_like(param),
			],
			output_modifiers=[lambda output: output],
			gradient_mapper=(lambda out_grad, outputs: [out_grad / sum(outputs)]),
			reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
		)
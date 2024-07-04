import h5py
import numpy as np
import torch

def load(path):
  f = h5py.File(path, mode='r')

  # tf:   conv3d kernel DxHxWxC_INxC_OUT
  # pyt:  conv3d kernel C_OUTxC_INxDxHxW

  return {
    'conv1.weight': torch.from_numpy(np.transpose(f['model_weights/3dconv_encoding_1/3dconv_encoding_1/kernel:0'][()], axes=(4, 3, 0, 1, 2))),
    'conv1.bias': torch.from_numpy(f['model_weights/3dconv_encoding_bias_1/3dconv_encoding_bias_1/bias:0'][()]),
    'down_conv1.weight': torch.from_numpy(np.transpose(f['model_weights/3dconv_encoding_pooling_1/3dconv_encoding_pooling_1/kernel:0'][()], axes=(4, 3, 0, 1, 2))),
    'down_conv1.bias': torch.from_numpy(f['model_weights/3dconv_encoding_pooling_bias_1/3dconv_encoding_pooling_bias_1/bias:0'][()]),

    'conv2.weight': torch.from_numpy(np.transpose(f['model_weights/3dconv_encoding_2/3dconv_encoding_2/kernel:0'][()], axes=(4, 3, 0, 1, 2))),
    'conv2.bias': torch.from_numpy(f['model_weights/3dconv_encoding_bias_2/3dconv_encoding_bias_2/bias:0'][()]),
    'down_conv2.weight': torch.from_numpy(np.transpose(f['model_weights/3dconv_encoding_pooling_2/3dconv_encoding_pooling_2/kernel:0'][()], axes=(4, 3, 0, 1, 2))),
    'down_conv2.bias': torch.from_numpy(f['model_weights/3dconv_encoding_pooling_bias_2/3dconv_encoding_pooling_bias_2/bias:0'][()]),

    'conv3.weight': torch.from_numpy(np.transpose(f['model_weights/3dconv_encoding_3/3dconv_encoding_3/kernel:0'][()], axes=(4, 3, 0, 1, 2))),
    'conv3.bias': torch.from_numpy(f['model_weights/3dconv_encoding_bias_3/3dconv_encoding_bias_3/bias:0'][()]),
    'down_conv3.weight': torch.from_numpy(np.transpose(f['model_weights/3dconv_encoding_pooling_3/3dconv_encoding_pooling_3/kernel:0'][()], axes=(4, 3, 0, 1, 2))),
    'down_conv3.bias': torch.from_numpy(f['model_weights/3dconv_encoding_pooling_bias_3/3dconv_encoding_pooling_bias_3/bias:0'][()]),

    'conv4.weight': torch.from_numpy(np.transpose(f['model_weights/3dconv_encoding_4/3dconv_encoding_4/kernel:0'][()], axes=(4, 3, 0, 1, 2))),
    'conv4.bias': torch.from_numpy(f['model_weights/3dconv_encoding_bias_4/3dconv_encoding_bias_4/bias:0'][()]),
    'down_conv4.weight': torch.from_numpy(np.transpose(f['model_weights/3dconv_encoding_pooling_4/3dconv_encoding_pooling_4/kernel:0'][()], axes=(4, 3, 0, 1, 2))),
    'down_conv4.bias': torch.from_numpy(f['model_weights/3dconv_encoding_pooling_bias_4/3dconv_encoding_pooling_bias_4/bias:0'][()]),

    'conv5.weight': torch.from_numpy(np.transpose(f['model_weights/3dconv_encoding_5/3dconv_encoding_5/kernel:0'][()], axes=(4, 3, 0, 1, 2))),
    'conv5.bias': torch.from_numpy(f['model_weights/3dconv_encoding_bias_5/3dconv_encoding_bias_5/bias:0'][()]),
    'down_conv5.weight': torch.from_numpy(np.transpose(f['model_weights/3dconv_encoding_pooling_5/3dconv_encoding_pooling_5/kernel:0'][()], axes=(4, 3, 0, 1, 2))),
    'down_conv5.bias': torch.from_numpy(f['model_weights/3dconv_encoding_pooling_bias_5/3dconv_encoding_pooling_bias_5/bias:0'][()]),

    'dens1.weight': torch.from_numpy(np.transpose(f['model_weights/dense_1/dense_1/kernel:0'][()], axes=(1, 0))),
    'dens1.bias': torch.from_numpy(f['model_weights/dense_bias_1/dense_bias_1/bias:0'][()]),
    
    'dens2.weight': torch.from_numpy(np.transpose(f['model_weights/dense_2/dense_2/kernel:0'][()], axes=(1, 0))),
    'dens2.bias': torch.from_numpy(f['model_weights/dense_bias_2/dense_bias_2/bias:0'][()]),
  }

from tensorflow.keras.layers import Input, Conv3D, Activation, Flatten, Dense

from layers import Bias
from constraints import Neg

def build_classifier(input_shape, filters, kernel_initializer):
  # Input
  input_tensor = Input(shape=input_shape, name='input')
  
  # Encoding 1
  encoded_1 = Conv3D(filters, (3, 3, 3),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     use_bias=False,
                     name='3dconv_encoding_1')(input_tensor)
  encoded_1 = Bias(bias_constraint=Neg(),
                   name='3dconv_encoding_bias_1')(encoded_1)
  encoded_1 = Activation('relu')(encoded_1)

  # Pooling 1
  pooled_1 = Conv3D(filters, (3, 3, 3),
                    strides=(2, 2, 2),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    use_bias=False,
                    name='3dconv_encoding_pooling_1')(encoded_1)
  pooled_1 = Bias(bias_constraint=Neg(),
                  name='3dconv_encoding_pooling_bias_1')(pooled_1)
  pooled_1 = Activation('relu')(pooled_1)

  # Encoding 2
  encoded_2 = Conv3D(filters, (3, 3, 3),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     use_bias=False,
                     name='3dconv_encoding_2')(pooled_1)
  encoded_2 = Bias(bias_constraint=Neg(),
                   name='3dconv_encoding_bias_2')(encoded_2)
  encoded_2 = Activation('relu')(encoded_2)

  # Pooling 2
  pooled_2 = Conv3D(filters, (3, 3, 3),
                    strides=(2, 2, 2),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    use_bias=False,
                    name='3dconv_encoding_pooling_2')(encoded_2)
  pooled_2 = Bias(bias_constraint=Neg(),
                  name='3dconv_encoding_pooling_bias_2')(pooled_2)
  pooled_2 = Activation('relu')(pooled_2)

  # Encoding 3
  encoded_3 = Conv3D(filters, (3, 3, 3),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     use_bias=False,
                     name='3dconv_encoding_3')(pooled_2)
  encoded_3 = Bias(bias_constraint=Neg(),
                   name='3dconv_encoding_bias_3')(encoded_3)
  encoded_3 = Activation('relu')(encoded_3)

  # Pooling 3
  pooled_3 = Conv3D(filters, (3, 3, 3),
                    strides=(2, 2, 2),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    use_bias=False,
                    name='3dconv_encoding_pooling_3')(encoded_3)
  pooled_3 = Bias(bias_constraint=Neg(),
                  name='3dconv_encoding_pooling_bias_3')(pooled_3)
  pooled_3 = Activation('relu')(pooled_3)

  # Encoding 4
  encoded_4 = Conv3D(filters, (3, 3, 3),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     use_bias=False,
                     name='3dconv_encoding_4')(pooled_3)
  encoded_4 = Bias(bias_constraint=Neg(),
                   name='3dconv_encoding_bias_4')(encoded_4)
  encoded_4 = Activation('relu')(encoded_4)

  # Pooling 4
  pooled_4 = Conv3D(filters, (3, 3, 3),
                    strides=(2, 2, 2),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    use_bias=False,
                    name='3dconv_encoding_pooling_4')(encoded_4)
  pooled_4 = Bias(bias_constraint=Neg(),
                  name='3dconv_encoding_pooling_bias_4')(pooled_4)
  pooled_4 = Activation('relu',
                        name='3dconv_encoding_pooling_activation_4')(pooled_4)

  # Encoding 5
  encoded_5 = Conv3D(filters, (3, 3, 3),
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     use_bias=False,
                     name='3dconv_encoding_5')(pooled_4)
  encoded_5 = Bias(bias_constraint=Neg(),
                   name='3dconv_encoding_bias_5')(encoded_5)
  encoded_5 = Activation('relu')(encoded_5)

  # Pooling 5
  pooled_5 = Conv3D(filters, (3, 3, 3),
                    strides=(2, 2, 2),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    use_bias=False,
                    name='3dconv_encoding_pooling_5')(encoded_5)
  pooled_5 = Bias(bias_constraint=Neg(),
                  name='3dconv_encoding_pooling_bias_5')(pooled_5)
  pooled_5 = Activation('relu',
                        name='3dconv_encoding_pooling_activation_5')(pooled_5)

  # Flatten
  flattened = Flatten()(pooled_5)

  # Dense 1
  densed_1 = Dense(16,
                   kernel_initializer=kernel_initializer,
                   use_bias=False,
                   name='dense_1')(flattened)
  densed_1 = Bias(bias_constraint=Neg(),
                  name='dense_bias_1')(densed_1)
  densed_1 = Activation('relu', 
                        name='dense_activation_1')(densed_1)

  # Dense 2
  densed_2 = Dense(2,
                   kernel_initializer=kernel_initializer,
                   use_bias=False,
                   name='dense_2')(densed_1)
  densed_2 = Bias(bias_constraint=Neg(),
                  name='dense_bias_2')(densed_2)
  densed_2 = Activation('softmax',
                        name='output')(densed_2)

  return (input_tensor, densed_2)

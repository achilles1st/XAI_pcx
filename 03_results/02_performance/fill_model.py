import h5py

from keras.saving import saving_utils
from keras.saving.saved_model import json_utils
from keras.saving import hdf5_format

def fill_model_from_hdf5(model, filepath, custom_objects=None, compile=True):
  if h5py is None:
    raise ImportError('`load_model()` using h5 format requires h5py. Could not '
                      'import h5py.')

  if not custom_objects:
    custom_objects = {}

  opened_new_file = not isinstance(filepath, h5py.File)
  if opened_new_file:
    f = h5py.File(filepath, mode='r')
  else:
    f = filepath

  try:
    # set weights
    hdf5_format.load_weights_from_hdf5_group(f['model_weights'], model)

    if compile:
      # instantiate optimizer
      training_config = f.attrs.get('training_config')
      if hasattr(training_config, 'decode'):
        training_config = training_config.decode('utf-8')
      if training_config is None:
        print('No training configuration found in the save file, so '
              'the model was *not* compiled. Compile it manually.')
        return
      training_config = json_utils.decode(training_config)

      # Compile model.
      model.compile(**saving_utils.compile_args_from_training_config(
          training_config, custom_objects), from_serialized=True)
      saving_utils.try_build_compiled_arguments(model)

      # Set optimizer weights.
      if 'optimizer_weights' in f:
        try:
          model.optimizer._create_all_weights(model.trainable_variables)
        except (NotImplementedError, AttributeError):
          print(
              'Error when creating the weights of optimizer {}, making it '
              'impossible to restore the saved optimizer state. As a result, '
              'your model is starting with a freshly initialized optimizer.')

        optimizer_weight_values = hdf5_format.load_optimizer_weights_from_hdf5_group(f)
        try:
          model.optimizer.set_weights(optimizer_weight_values)
        except ValueError:
          print('Error in loading the saved optimizer '
                'state. As a result, your model is '
                'starting with a freshly initialized '
                'optimizer.')
  finally:
    if opened_new_file:
      f.close()

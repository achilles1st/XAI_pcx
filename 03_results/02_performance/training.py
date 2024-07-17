import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from model import build_classifier
from deep_taylor_decomposition import add_deep_taylor_decomposition_to_model_output
from losses import relevance_guided
from metrics import sum_relevance_inner_mask, sum_relevance
from sequence import InputSequence
from callbacks import ReduceLRAndResetWeightsOnPlateau
from fill_model import fill_model_from_hdf5

def training(last_saved_model, bootstrap_index, initial_weights_index, start_epoch, training_config, training_data_paths, validation_data_paths, batch_size, epochs):
  (learning_rate, reduce_factor, input_image, input_mask, use_bet, use_rg, input_shape, params_destination, _, path_to_initial_weights) = training_config

  K.clear_session()
  data_intensity_normalization_constant = 40.

  # classifier
  (input_tensor, output_tensor) = build_classifier(input_shape + (1,), filters=8, kernel_initializer='he_uniform') 
    
  if use_rg:
    # relevance-guided extension
    heatmap_tensor = add_deep_taylor_decomposition_to_model_output(output_tensor)
    model = Model(inputs=input_tensor, outputs=[output_tensor, heatmap_tensor])
  else:
    # only classifier
    model = Model(inputs=input_tensor, outputs=output_tensor)

  # continue
  if last_saved_model is not None:
    fill_model_from_hdf5(
      model,
      os.path.join('weights', params_destination, last_saved_model),
      custom_objects={
        'relevance_guided': relevance_guided,
        'sum_relevance_inner_mask': sum_relevance_inner_mask,
        'sum_relevance': sum_relevance,
      },
      compile=True,
    )
  # start new
  else:
    losses = []
    metrics = {}

    if use_rg:
      # losses
      losses.append(tf.keras.losses.CategoricalCrossentropy())
      losses.append(relevance_guided)
      
      # metrics
      metrics = {
        'output': [tf.keras.metrics.CategoricalAccuracy()],
        'input_dtd': [sum_relevance_inner_mask, sum_relevance],
      }
    else:
      # losses
      losses.append(tf.keras.losses.CategoricalCrossentropy())

      # metrics
      metrics = {
        'output': [tf.keras.metrics.CategoricalAccuracy()]
      }
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
      optimizer=optimizer,
      loss=losses,
      metrics=metrics,
    )

    if path_to_initial_weights is not None:
      final_path_to_initial_weights = path_to_initial_weights.format(initial_weights_index)
      print(final_path_to_initial_weights)
      model.load_weights(final_path_to_initial_weights)

  training_sequence = InputSequence(
    training_data_paths,
    [
      input_image,
    ],
    [
      input_mask,
    ],
    input_shape,
    data_intensity_normalization_constant,
    batch_size=batch_size,
    use_shuffle=True,
    output_only_categories=not use_rg,
    use_mask_on_input=use_bet,
  )

  validation_sequence = InputSequence(
    validation_data_paths,
    [
      input_image,
    ],
    [
      input_mask,
    ],
    input_shape,
    data_intensity_normalization_constant,
    batch_size=batch_size,
    use_shuffle=False,
    output_only_categories=not use_rg,
    use_mask_on_input=use_bet,
  )

  model.summary()

  model.fit(
    x=training_sequence,
    epochs=epochs,
    validation_data=validation_sequence,
    callbacks=[
      ModelCheckpoint(
        os.path.join(
          'weights',
          params_destination,
          'R2star__boostrap_index-{0:02d}__initial_weights_index-{1:02d}'.format(bootstrap_index, initial_weights_index) + '__{epoch:03d}-tcl-{output_loss:.3f}-vcl-{val_output_loss:.3f}-tca-{output_categorical_accuracy:.3f}-vca-{val_output_categorical_accuracy:.3f}-tsrim-{input_dtd_sum_relevance_inner_mask:.3f}-vsrim-{val_input_dtd_sum_relevance_inner_mask:.3f}.h5'
          if use_rg
          else 'R2star__boostrap_index-{0:02d}__initial_weights_index-{1:02d}'.format(bootstrap_index, initial_weights_index) + '__{epoch:03d}-tcl-{loss:.3f}-vcl-{val_loss:.3f}-tca-{categorical_accuracy:.3f}-vca-{val_categorical_accuracy:.3f}.h5'
        ),
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=False,
      ),
      ReduceLRAndResetWeightsOnPlateau(
        monitor='val_loss',
        factor=reduce_factor,
        patience=5,
        mode='auto',
        min_delta=1e-4,
        cooldown=0,
        min_lr=1e-6,
        optimizer=model.optimizer,
        verbose=1,
      ),
    ],
    initial_epoch=start_epoch,
    max_queue_size=2,
    workers=2,
    use_multiprocessing=False,
  )

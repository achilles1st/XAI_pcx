import tensorflow as tf
import numpy as np

class ReduceLRAndResetWeightsOnPlateau(tf.keras.callbacks.Callback):
  def __init__(self,
               monitor='val_loss',
               factor=0.1,
               patience=10,
               verbose=0,
               mode='auto',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0,
               optimizer=None,
               **kwargs):
    super(ReduceLRAndResetWeightsOnPlateau, self).__init__()

    self.monitor = monitor
    if factor >= 1.0:
      raise ValueError(f'ReduceLROnPlateau does not support a factor >= 1.0. Got {factor}')
    self.factor = factor
    self.min_lr = min_lr
    self.min_delta = min_delta
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.optimizer = optimizer
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.best = 0
    self.weights = None
    self.mode = mode
    self.monitor_op = None
    self._reset()

  def _reset(self):
    """Resets wait counter and cooldown counter.
    """
    if self.mode not in ['auto', 'min', 'max']:
      print('Learning rate reduction mode %s is unknown, '
            'fallback to auto mode.', self.mode)
      self.mode = 'auto'
    if (self.mode == 'min' or
        (self.mode == 'auto' and 'acc' not in self.monitor)):
      self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
      self.best = np.Inf
    else:
      self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
      self.best = -np.Inf
    self.cooldown_counter = 0
    self.wait = 0

  def set_model(self, model):
    super(ReduceLRAndResetWeightsOnPlateau, self).set_model(model)
    self.weights = self.model.get_weights()

  def on_train_begin(self, logs=None):
    self._reset()

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = tf.keras.backend.get_value(self.optimizer.lr)
    current = logs.get(self.monitor)
    if current is None:
      print('Learning rate reduction is conditioned on metric `%s` '
            'which is not available. Available metrics are: %s',
            self.monitor, ','.join(list(logs.keys())))
    else:
      if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0

      if self.monitor_op(current, self.best):
        self.best = current
        self.weights = self.model.get_weights()
        self.wait = 0
      elif not self.in_cooldown():
        self.wait += 1
        if self.wait >= self.patience:
          old_lr = tf.keras.backend.get_value(self.optimizer.lr)
          if old_lr > np.float32(self.min_lr):
            new_lr = old_lr * self.factor
            new_lr = max(new_lr, self.min_lr)
            tf.keras.backend.set_value(self.optimizer.lr, new_lr)
            if self.verbose > 0:
              print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                    'rate to %s.' % (epoch + 1, new_lr))
            if self.weights is not None:
              self.model.set_weights(self.weights)
              if self.verbose > 0:
                print('\nEpoch %05d: Reset weights.' % (epoch + 1))
            self.cooldown_counter = self.cooldown
            self.wait = 0

  def in_cooldown(self):
    return self.cooldown_counter > 0

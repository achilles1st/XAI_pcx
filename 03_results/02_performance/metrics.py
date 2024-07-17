from tensorflow.keras import backend as K

def sum_relevance_inner_mask(y_true, y_pred):
  return K.sum(y_true * y_pred, axis=[1, 2, 3, 4])

def sum_relevance(y_true, y_pred):
  return K.sum(y_pred, axis=[1, 2, 3, 4])

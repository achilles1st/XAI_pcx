from tensorflow.keras import backend as K

def relevance_guided(y_true, y_pred):
  return -K.sum(y_true * y_pred, axis=[1, 2, 3, 4])

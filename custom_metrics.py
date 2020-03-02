from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def f1_score(y_true, y_pred):
    """ Calculates the f1 score using average=macro. Calculate metrics for each label, and find
    their unweighted mean. This does not take label imbalance into account.
    Code inspired by https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras """

    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)

    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(f1 == np.nan, tf.zeros_like(f1), f1)
    return K.mean(f1)

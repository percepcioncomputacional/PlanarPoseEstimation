from keras import backend as K
import tfquaternion as tfq
import tensorflow as tf
import settings


def euc_loss1x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    return 0.3 * lx


def euc_loss2x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    return 0.3 * lx


def euc_loss3x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    return lx


def euc_loss1q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    return 0.3 * settings.BETA * lq


def euc_loss2q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    return 0.3 * settings.BETA * lq


def euc_loss3q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    return settings.BETA * lq


def frob_loss1r(y_true, y_pred):
    y_true = K.reshape(y_true, [-1, y_pred.shape[-1]])
    y_true = tfq.Quaternion(y_true)
    m_true = y_true.as_rotation_matrix()

    y_pred = tfq.Quaternion(y_pred)
    m_pred = y_pred.as_rotation_matrix()
    return 0.3 * settings.BETA * tf.norm((m_true - m_pred), ord='fro', axis=[-2, -1], keepdims=True)


def frob_loss2r(y_true, y_pred):
    y_true = K.reshape(y_true, [-1, y_pred.shape[-1]])
    y_true = tfq.Quaternion(y_true)
    m_true = y_true.as_rotation_matrix()

    y_pred = tfq.Quaternion(y_pred)
    m_pred = y_pred.as_rotation_matrix()
    return 0.3 * settings.BETA * tf.norm((m_true - m_pred), ord='fro', axis=[-2, -1], keepdims=True)


def frob_loss3r(y_true, y_pred):
    y_true = K.reshape(y_true, [-1, y_pred.shape[-1]])
    y_true = tfq.Quaternion(y_true)
    m_true = y_true.as_rotation_matrix()

    y_pred = tfq.Quaternion(y_pred)
    m_pred = y_pred.as_rotation_matrix()
    return settings.BETA * tf.norm((m_true - m_pred), ord='fro', axis=[-2, -1], keepdims=True)



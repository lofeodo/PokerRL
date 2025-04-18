'''
	Computes a Huber loss for neural net training and evaluation.
	Computes the loss across buckets.
'''

import tensorflow as tf


# main loss
class BasicHuberLoss(tf.keras.losses.Loss):
	def __init__(self, delta=1.0, name='basic_huber_loss', reduction=tf.keras.losses.Reduction.AUTO):
		super().__init__(name=name, reduction=reduction)
		self.delta = delta

	def call(self, y_true, y_pred):
		error = y_true - y_pred
		abs_error = tf.abs(error)
		quadratic = tf.minimum(abs_error, self.delta)
		linear = abs_error - quadratic
		return tf.reduce_mean(0.5 * tf.square(quadratic) + self.delta * linear)


# used only as metric
def masked_huber_loss(y_true, y_pred):
	error = y_true - y_pred
	abs_error = tf.abs(error)
	quadratic = tf.minimum(abs_error, 1.0)
	linear = abs_error - quadratic
	loss = 0.5 * tf.square(quadratic) + 1.0 * linear
	return tf.reduce_mean(loss)
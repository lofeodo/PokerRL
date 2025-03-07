'''
	Trains the neural network
'''
import os
import random
random.seed(9999)
import numpy as np
import tensorflow as tf

from Settings.arguments import arguments
from NnTraining.tf_data import create_iterator
from NeuralNetwork.value_nn import ValueNn
from NeuralNetwork.metrics import BasicHuberLoss, masked_huber_loss

class Train(ValueNn):
	def __init__(self, data_dir_list, street):
		'''
		@param: [str,...] :list of paths to directories that contains tf records
		'''
		# set up estimator from ValueNn
		super().__init__(street)
		
		# set up read paths for train/valid datasets
		self.tfrecords = [f.path for dirpath in data_dir_list for f in os.scandir(dirpath)]
		self.create_keras_callback()


	def compile_keras_model(self, keras_model):
		''' compiles keras model '''
		loss = BasicHuberLoss(delta=1.0)
		optimizer = tf.keras.optimizers.Adam(
			learning_rate=arguments.learning_rate,
			beta_1=0.9,
			beta_2=0.999
		)
		keras_model.compile(
			loss=loss,
			optimizer=optimizer,
			metrics=[masked_huber_loss],
			run_eagerly=True
		)


	def train(self, num_epochs, batch_size, verbose=1, validation_size=0.1, start_epoch=0):
		''' trains model with specified parameters '''
		# Compile model first
		self.compile_keras_model(self.keras_model)
		
		# Use all files for both training and validation
		train_filenames = self.tfrecords
		valid_filenames = self.tfrecords  # Use same files
		
		# create tf.data iterators with internal splitting
		train_iterator = create_iterator(filenames=train_filenames, 
									   train=True,
									   batch_size=batch_size,
									   x_shape=self.x_shape, 
									   y_shape=self.y_shape,
									   validation_size=validation_size,
									   is_validation=False)
		
		valid_iterator = create_iterator(filenames=valid_filenames, 
									   train=False,
									   batch_size=batch_size,
									   x_shape=self.x_shape, 
									   y_shape=self.y_shape,
									   validation_size=validation_size,
									   is_validation=True)

		# count num of elements in both sets
		num_train_elements = len(train_filenames) * arguments.tfrecords_batch_size
		num_valid_elements = len(valid_filenames) * arguments.tfrecords_batch_size
		
		# Convert iterators to tf.data.Dataset with dynamic batch sizes
		train_dataset = tf.data.Dataset.from_generator(
			lambda: train_iterator,
			output_signature=(
				tf.TensorSpec(shape=(None, *self.x_shape), dtype=tf.float32),
				tf.TensorSpec(shape=(None, *self.y_shape), dtype=tf.float32) 
			)
		).prefetch(tf.data.AUTOTUNE)
		
		valid_dataset = tf.data.Dataset.from_generator(
			lambda: valid_iterator,
			output_signature=(
				tf.TensorSpec(shape=(None, *self.x_shape), dtype=tf.float32),
				tf.TensorSpec(shape=(None, *self.y_shape), dtype=tf.float32) 
			)
		).prefetch(tf.data.AUTOTUNE)
		
		h = self.keras_model.fit(
			train_dataset,
			validation_data=valid_dataset,
			steps_per_epoch=num_train_elements // batch_size,
			validation_steps=num_valid_elements // batch_size,
			epochs=num_epochs,
			verbose=verbose,
			callbacks=self.callbacks,
			initial_epoch=start_epoch
		)


	def create_keras_callback(self):
		''' returns list of keras callbacks '''
		# imports
		from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, \
												  LearningRateScheduler, EarlyStopping
		# tensorboard callback
		tb_logdir = os.path.join( self.model_dir_path, 'tensorboard' )
		tb = tf.keras.callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=25, write_graph=True)
		# Early stopping callback
		es = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
		# Save model callback
		mc = ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss', mode='min')
		# Change learning rate
		lrs = LearningRateScheduler( lambda epoch: arguments.learning_rate )
		# Reducting LR callback
		lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_delta=1e-4, mode='min')
		# set keras callback for training
		self.callbacks = [tb, mc, lrs]


class KerasTensorBoard(tf.keras.callbacks.TensorBoard):
	''' keras callback, that saves training/validation losses to tensorboard '''
	def __init__(self, log_dir, **kwargs):
		# Make the original `TensorBoard` log to a subdirectory 'training'
		training_log_dir = os.path.join(log_dir, 'training')
		super(KerasTensorBoard, self).__init__(training_log_dir, **kwargs)
		# Log the validation metrics to a separate subdirectory
		self.val_log_dir = os.path.join(log_dir, 'validation')

	def set_model(self, model):
		# Setup writer for validation metrics
		self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
		super(KerasTensorBoard, self).set_model(model)

	def on_epoch_end(self, epoch, logs=None):
		# Pop the validation logs and handle them separately with
		# `self.val_writer`. Also rename the keys so that they can
		# be plotted on the same figure with the training metrics
		logs = logs or {}
		val_logs = {k.replace('val_', 'epoch_'): v for k, v in logs.items() if k.startswith('val_')}
		with self.val_writer.as_default():
			for name, value in val_logs.items():
				tf.summary.scalar(name, value, step=epoch)
		# Pass the remaining logs to `TensorBoard.on_epoch_end`
		logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
		super(KerasTensorBoard, self).on_epoch_end(epoch, logs)

	def on_train_end(self, logs=None):
		super(KerasTensorBoard, self).on_train_end(logs)
		self.val_writer.close()




#

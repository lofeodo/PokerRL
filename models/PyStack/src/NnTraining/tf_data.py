'''
	Helper functions for reading and parsing data from TFRecords
'''
import os
import tensorflow as tf


def create_parse_fn(x_shape, y_shape):
	''' Creates parse function for tf.data.TFRecordDataset
	@param: [1] :x shape (not including batch size) ex: [224,224,3] if img
	@param: [1] :y shape (not including batch size) ex: [224,224,3] if img
	@return parse function
	'''
	def parse_fn(serialized):
		# Define a dict with the data-names and types we expect to
		# find in the TFRecords file.
		features = {
					'input': tf.io.FixedLenFeature([], tf.string),
					'output': tf.io.FixedLenFeature([], tf.string)
				   }
		# Parse the serialized data so we get a dict with our data.
		parsed_example = tf.io.parse_single_example(serialized=serialized,
												  features=features)
		# Get the image as raw bytes.
		x_raw = parsed_example['input']
		y_raw = parsed_example['output']
		# m_raw = parsed_example['mask']
		# Decode the raw bytes so it becomes a tensor with type.
		x = tf.io.decode_raw(x_raw, tf.float32)
		y = tf.io.decode_raw(y_raw, tf.float32)
		# m = tf.decode_raw(m_raw, tf.uint8)
		# apply transormations
		# m = tf.cast(m, tf.float32)
		# # repeat mask 2 times ex: (36,) -> (72,)
		# y = y * m
		# x = x * m
		# apply shape
		x = tf.reshape(x, x_shape)
		y = tf.reshape(y, y_shape)
		# return
		return x, y
	return parse_fn


def create_iterator(filenames, train, x_shape, y_shape, batch_size, validation_size=0.1, is_validation=False, num_cores=os.cpu_count()):
	'''
	@param: [str,...] :Filenames for the TFRecords files.
	@param: bool      :Boolean whether training (True) or testing (False).
	@param: [1]       :input  shape (not including batch size) ex: [224,224,3] if img
	@param: [1]       :output shape (not including batch size) ex: [224,224,3] if img
	@param: int       :return batches of this size.
	@param: float     :fraction of data to use for validation
	@param: bool      :whether this iterator is for validation data
	@param: int       :number of cores to use
	@return tf.data.Dataset object
	'''
	# Load dataset
	dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=num_cores)
	
	# Count total number of elements
	total_elements = sum(1 for _ in dataset)
	print(f"Total elements in dataset: {total_elements}")
	
	# Calculate validation split size based on actual data
	val_size = int(validation_size * total_elements)
	print(f"Validation size: {val_size}")
	
	# Reset dataset after counting
	dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=num_cores)
	
	# Add deterministic shuffle with smaller buffer for small dataset
	dataset = dataset.shuffle(buffer_size=total_elements, seed=42)
	
	# Split based on actual data size
	if is_validation:
		dataset = dataset.take(val_size)
	else:
		dataset = dataset.skip(val_size)
	
	if train:
		dataset = dataset.shuffle(buffer_size=total_elements,
								reshuffle_each_iteration=True)
	
	# Parse the serialized data and create batches
	dataset = dataset.map(
		create_parse_fn(x_shape, y_shape),
		num_parallel_calls=tf.data.AUTOTUNE
	)
	
	# Use drop_remainder=False to handle partial batches
	dataset = dataset.batch(batch_size, drop_remainder=False)
	
	# Only repeat after batching for small datasets
	dataset = dataset.repeat()
	
	# prefetch for better performance
	dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
	
	return iter(dataset)




#

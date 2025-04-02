# -*- coding: utf-8 -*-
"""Wide Residual Network models for Keras.

# Reference

- [Wide Residual Networks](https://arxiv.org/abs/1605.07146)

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Activation, MaxPooling2D, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.layers import Add as add
from tensorflow.keras.utils import get_source_inputs
import tensorflow.keras.backend as K
import tensorflow as tf

def __conv1_block(input):
	x = Conv2D(16, (3, 3), padding='same')(input)

	channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)
	return x


def __conv2_block(input, k=1, dropout=0.0):
	init = input

	channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

	# Check if input number of filters is same as 16 * k, else create
	# convolution2d for this input
	if K.image_data_format() == 'channels_first':
		if init.shape[1] != 16 * k:
			init = Conv2D(16 * k, (1, 1), activation='linear', padding='same')(init)
	else:
		if init.shape[-1] != 16 * k:
			init = Conv2D(16 * k, (1, 1), activation='linear', padding='same')(init)

	x = Conv2D(16 * k, (3, 3), padding='same')(input)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	if dropout > 0.0:
		x = Dropout(dropout)(x)

	x = Conv2D(16 * k, (3, 3), padding='same')(x)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	m = add()([init, x])
	return m


def __conv3_block(input, k=1, dropout=0.0):
	init = input

	channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

	# Check if input number of filters is same as 32 * k, else
	# create convolution2d for this input
	if K.image_data_format() == 'channels_first':
		if init.shape[1] != 32 * k:
			init = Conv2D(32 * k, (1, 1), activation='linear', padding='same')(init)
	else:
		if init.shape[-1] != 32 * k:
			init = Conv2D(32 * k, (1, 1), activation='linear', padding='same')(init)

	x = Conv2D(32 * k, (3, 3), padding='same')(input)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	if dropout > 0.0:
		x = Dropout(dropout)(x)

	x = Conv2D(32 * k, (3, 3), padding='same')(x)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	m = add()([init, x])
	return m


def ___conv4_block(input, k=1, dropout=0.0):

	with tf.name_scope('Conv4'):
		init = input

		channel_axis = 1 if K.image_data_format() == 'th' else -1

		# Check if input number of filters is same as 64 * k, else
		# create convolution2d for this input
		if K.image_data_format() == 'th':
			if init.shape[1] != 64 * k:
				init = Conv2D(64 * k, (1, 1), activation='linear', padding='same')(init)
		else:
			if init.shape[-1] != 64 * k:
				init = Conv2D(64 * k, (1, 1), activation='linear', padding='same')(init)

		x = Conv2D(64 * k, (3, 3), padding='same')(input)
		x = BatchNormalization(axis=channel_axis)(x)
		x = Activation('relu')(x)

		if dropout > 0.0:
			x = Dropout(dropout)(x)

		x = Conv2D(64 * k, (3, 3), padding='same')(x)
		x = BatchNormalization(axis=channel_axis)(x)
		x = Activation('relu')(x)

		m = add()([init, x])
	return m


def create_wide_residual_network(nb_classes, img_input, include_top, depth=28,
								   width=8, dropout=0.0, activation='softmax'):
	''' Creates a Wide Residual Network with specified parameters

	Args:
		nb_classes: Number of output classes
		img_input: Input tensor or layer
		include_top: Flag to include the last dense layer
		depth: Depth of the network. Compute N = (n - 4) / 6.
			   For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
			   For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
			   For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
		width: Width of the network.
		dropout: Adds dropout if value is greater than 0.0

	Returns:a Keras Model
	'''

	N = (depth - 4) // 6

	x = __conv1_block(img_input)
	nb_conv = 4

	for i in range(N):
		x = __conv2_block(x, width, dropout)
		nb_conv += 2

	x = MaxPooling2D((2, 2))(x)

	for i in range(N):
		x = __conv3_block(x, width, dropout)
		nb_conv += 2

	x = MaxPooling2D((2, 2))(x)

	for i in range(N):
		x = ___conv4_block(x, width, dropout)
		nb_conv += 2

	if include_top:
		x = GlobalAveragePooling2D()(x)
		x = Dense(nb_classes, activation=activation)(x)

	return x

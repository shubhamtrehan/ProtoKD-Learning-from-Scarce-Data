from wide_resnet import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

def make_feat_extractor(input_shape=(32,32,3), num_classes=10, depth=28, width=8, dropout_rate=0.1, model_name='wide-resnet', top=True):
	'''
	Args:
		input_shape: (img_size, img_size, n_channels)
		num_classes: Number of classes in output layer
		depth: WResNet depth
		width: WResNet width
	'''
	img_input = Input(shape=input_shape)
	img_input = tf.keras.applications.resnet50.preprocess_input(img_input)
	feats = create_wide_residual_network(nb_classes=num_classes, img_input=img_input, include_top=False, depth=depth, width=width, dropout=dropout_rate, activation=None)
	
	feats = GlobalAveragePooling2D()(feats)
	logits = Dense(num_classes, activation='softmax', name="Logits_Layer")(feats)
	model = Model(img_input, [feats, logits], name=model_name)
	return model



from __future__ import division
import numpy as np
import theano.tensor as T

from keras.engine.topology import Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.recurrent import Recurrent
from keras import initializations
from keras import activations
from keras import backend as K
import keras
from keras.models import Sequential, Model
from keras.layers import Input, ConvLSTM2D, Flatten, Dense, Merge, merge
from keras.layers.core import Reshape
#import extra 
from keras.engine.topology import Layer
from keras.layers.wrappers import TimeDistributed
import os, scipy.misc
from moviepy.editor import *

class Normalize(Layer):
    '''
    Custom layer to normalize the input
    '''
    def __init__(self, **kwargs):
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return (x / 5.0)

    def get_output_shape_for(self, input_shape):
        return input_shape


def LSTM_RCN(include_top=True, weights='imagenet',
		  input_tensor=None, input_shape=None,
		  classes=1000):

	img_input = Input(shape=(10, 3, 224, 224))

	# Block 1
	td_conv1_1 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1'))(img_input)
	td_conv1_2 = TimeDistributed(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))(td_conv1_1)
	td_pool1   = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(td_conv1_2)

	# Block 2
	td_conv2_1 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))(td_pool1)
	td_conv2_2 = TimeDistributed(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))(td_conv2_1)
	td_pool2   = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(td_conv2_2)

	# Block 3
	td_conv3_1 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))(td_pool2)
	td_conv3_2 = TimeDistributed(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))(td_conv3_1)
	td_conv3_3 = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))(td_conv3_2)
	td_pool3   = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(td_conv3_3)

	# Block 4
	td_conv4_1 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))(td_pool3)
	td_conv4_2 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))(td_conv4_1)
	td_conv4_3 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))(td_conv4_2)
	td_pool4   = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(td_conv4_3)

	# Block 5
	td_conv5_1 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))(td_pool4)
	td_conv5_2 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))(td_conv5_1)
	td_conv5_3 = TimeDistributed(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))(td_conv5_2)
	td_pool5   = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(td_conv5_3)

	x   = TimeDistributed(Flatten(name='flatten'))(td_pool5)
	fc7 = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(x)
	fc7 = TimeDistributed(Reshape((4096, 1, 1)))(fc7)

	gru_rcn1 = ConvLSTM2D(64, 3, 3,border_mode='same')(td_pool2)
	gru_rcn2 = ConvLSTM2D(128, 3, 3,border_mode='same')(td_pool3)
	gru_rcn3 = ConvLSTM2D(256, 3, 3,border_mode='same')(td_pool4)
	gru_rcn4 = ConvLSTM2D(512, 3, 3,border_mode='same')(td_pool5)
	gru_rcn5 = ConvLSTM2D(1024, 3, 3, border_mode='same')(fc7)

	pool_rcn1 = AveragePooling2D((56, 56))(gru_rcn1)
	pool_rcn2 = AveragePooling2D((28, 28))(gru_rcn2)
	pool_rcn3 = AveragePooling2D((14, 14))(gru_rcn3)
	pool_rcn4 = AveragePooling2D((7, 7))(gru_rcn4)
	pool_rcn5 = AveragePooling2D((1, 1))(gru_rcn5)

	flat_rcn1 = Flatten()(pool_rcn1)
	flat_rcn2 = Flatten()(pool_rcn2)
	flat_rcn3 = Flatten()(pool_rcn3)
	flat_rcn4 = Flatten()(pool_rcn4)
	flat_rcn5 = Flatten()(pool_rcn5)

	dense_c1 = Dense(101, activation='softmax')(flat_rcn1)
	dense_c2 = Dense(101, activation='softmax')(flat_rcn2)
	dense_c3 = Dense(101, activation='softmax')(flat_rcn3)	
	dense_c4 = Dense(101, activation='softmax')(flat_rcn4)
	dense_c5 = Dense(101, activation='softmax')(flat_rcn5)

	merged_model = merge([dense_c1, dense_c2, dense_c3, dense_c4, dense_c5], mode='sum')

	output = Normalize()(merged_model)
	model = Model(img_input, output)
	print model.summary()

	return model

class GRU_RCN(Recurrent):
	"""RNN with all connections being convolutions:
	H_t = activation(conv(H_tm1, W_hh) + conv(X_t, W_ih) + b)
	with H_t and X_t being images and W being filters.
	We use Keras' RNN API, thus input and outputs should be 3-way tensors.
	Assuming that your input video have frames of size
	[nb_channels, nb_rows, nb_cols], the input of this layer should be reshaped
	to [batch_size, time_length, nb_channels*nb_rows*nb_cols]. Thus, you have to
	pass the original images shape to the ConvRNN layer.

	Parameters:
	-----------
	filter_dim: list [nb_filters, nb_row, nb_col] convolutional filter
		dimensions
	reshape_dim: list [nb_channels, nb_row, nb_col] original dimensions of a
		frame.
	batch_size: int, batch_size is useful for TensorFlow backend.
	time_length: int, optional for Theano, mandatory for TensorFlow
	subsample: (int, int), just keras.layers.Convolutional2D.subsample
	"""
	def __init__(self, filter_dim, reshape_dim,
				 batch_size=None, subsample=(1, 1),
				 init='glorot_uniform', inner_init='glorot_uniform',
				 activation='tanh', inner_activation='hard_sigmoid',
				 use_previous=False,
				 weights=None, **kwargs):
		self.batch_size = batch_size
		self.border_mode = 'same'
		self.filter_dim = filter_dim
		self.reshape_dim = reshape_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.initial_weights = weights

		self.use_previous = use_previous
		self.subsample = tuple(subsample)
		self.output_dim = (filter_dim[0], reshape_dim[1], reshape_dim[2])
		super(GRU_RCN, self).__init__(**kwargs)

	def _get_batch_size(self, X):
		if K._BACKEND == 'theano':
			batch_size = X.shape[0]
		else:
			batch_size = self.batch_size
		return batch_size

	def build(self, input_shape):
		if K._BACKEND == 'theano':
			batch_size = None
		else:
			batch_size = None  # self.batch_size
		bm = self.border_mode
		reshape_dim = self.reshape_dim
		hidden_dim = self.output_dim

		nb_filter, nb_rows, nb_cols = self.filter_dim

		self.b_h = K.zeros((nb_filter,))
		self.b_r = K.zeros((nb_filter,))
		self.b_z = K.zeros((nb_filter,))

		self.conv_h = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)
		self.conv_z = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)
		self.conv_r = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)

		self.conv_x_h = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)
		self.conv_x_z = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)
		self.conv_x_r = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)

		# hidden to hidden connections
		self.conv_h.build(hidden_dim)
		self.conv_z.build(hidden_dim)
		self.conv_r.build(hidden_dim)
		# input to hidden connections
		self.conv_x_h.build(reshape_dim)
		self.conv_x_z.build(reshape_dim)
		self.conv_x_r.build(reshape_dim)

		self.trainable_weights = self.conv_h.trainable_weights + self.conv_z.trainable_weights + self.conv_r.trainable_weights + \
			self.conv_x_h.trainable_weights + self.conv_x_z.trainable_weights + self.conv_x_r.trainable_weights

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def get_initial_states(self, X):
		batch_size = self._get_batch_size(X)
		hidden_dim = np.prod(self.output_dim)
		if K._BACKEND == 'theano':
			h = T.zeros((batch_size, hidden_dim))
		else:
			h = K.zeros((batch_size, hidden_dim))
		return [h, ]

	def step(self, x, states):
		batch_size = self._get_batch_size(x)
		input_shape = (batch_size, ) + self.reshape_dim
		hidden_dim = (batch_size, ) + self.output_dim
		nb_filter, nb_rows, nb_cols = self.output_dim
		h_tm1 = K.reshape(states[0], hidden_dim)

		x_t = K.reshape(x, input_shape)
		xz_t = self.conv_x_z(x_t)
		xr_t = self.conv_x_r(x_t)
		xh_t = self.conv_x_h(x_t)

		z = self.inner_activation(xz_t + self.conv_z(h_tm1))
		r = self.inner_activation(xr_t + self.conv_r(h_tm1))

		hh_t = self.activation(xh_t + self.conv_h(r * h_tm1))
		h_t = z * h_tm1 + (1 - z) * hh_t
		h_t = K.batch_flatten(h_t)
		return h_t, [h_t, ]

	@property
	def output_shape(self):
		input_shape = self.input_shape
		if self.return_sequences:
			return (input_shape[0], input_shape[1], np.prod(self.output_dim))
		else:
			return (input_shape[0], np.prod(self.output_dim))

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "filter_dim": self.filter_dim,
				  "output_dim": self.output_dim,
				  "init": self.init.__name__,
				  "inner_init": self.inner_init.__name__,
				  "activation": self.activation.__name__,
				  "inner_activation": self.inner_activation.__name__,
				  "return_sequences": self.return_sequences,
				  "reshape_dim": self.reshape_dim,
				  "go_backwards": self.go_backwards}
		base_config = super(ConvGRU, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def read_video(filename):
	video = VideoFileClip('../../../UCF/' + filename)
	img_list = []
	
	for frame in video.iter_frames():
		img_list.append(np.transpose(scipy.misc.imresize(frame, (224, 224, 3)), (2, 0, 1)))
	img_list = np.asarray(img_list)

	indexed = np.int32(np.linspace(0, len(img_list) - 1, 10))
	return img_list[indexed]

if __name__=='__main__':
	#print ('in main')
	model = LSTM_RCN()

	train_data = []
	# Read the training files
	with open('../../../UCF/ucfTrainTestlist/trainlist01.txt') as f:
		data = f.readlines()
		for row in data:
			row = row.split(' ')
			train_data.append([row[0].split('/')[-1], int(row[1].replace('\r\n', ''))])

	with open('../../../UCF/ucfTrainTestlist/trainlist02.txt') as f:
		data = f.readlines()
		for row in data:
			row = row.split(' ')
			train_data.append([row[0].split('/')[-1], int(row[1].replace('\r\n', ''))])

	with open('../../../UCF/ucfTrainTestlist/trainlist03.txt') as f:
		data = f.readlines()
		for row in data:
			row = row.split(' ')
			train_data.append([row[0].split('/')[-1], int(row[1].replace('\r\n', ''))])


	test_data = []
	# Read the test files
	with open('../../../UCF/ucfTrainTestlist/testlist01.txt') as f:
		data = f.readlines()
		for row in data:
			test_data.append(row.replace('\r\n', '').split('/')[-1])

	with open('../../../UCF/ucfTrainTestlist/testlist02.txt') as f:
		data = f.readlines()
		for row in data:
			test_data.append(row.replace('\r\n', '').split('/')[-1])

	with open('../../../UCF/ucfTrainTestlist/testlist03.txt') as f:
		data = f.readlines()
		for row in data:
			test_data.append(row.replace('\r\n', '').split('/')[-1])

	print (len(test_data), len(train_data))

	trainX, trainY = [], []
	for data in train_data:
		if '.avi' not in data[0]:
			continue
		trainX.append(read_video(data[0]))
		trainY.append(data[1])	
		if len(trainX) == 10:
			break
		print ('here')
	
	trainX = np.asarray(trainX)
	trainY = np.asarray(trainY)
	
	print (trainX.shape, trainY.shape)
	model.compile(optimizer='adam',
              	  loss='categorical_crossentropy',
              	  metrics=['accuracy'])
	model.fit(trainX, keras.utils.np_utils.to_categorical(trainY, 101), nb_epoch=10, batch_size=2)
	model.save('my_model.h5')
#VGG16(input_shape=(3, 224, 224))

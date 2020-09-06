import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle
from tensorflow.python.ops import rnn_cell
import tensorflow.python.platform
import csv
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from skimage import io, transform
import random, os
import scipy.misc as misc
import cPickle as pickle
from nltk.tokenize import RegexpTokenizer
from moviepy.editor import *
from keras import backend as K
import h5py

import math
import os
#import ipdb
tf.python.control_flow_ops = tf

# with tf.device('/gpu:0'):
def VGG_16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	vgg_out = K.function([model.layers[0].input, K.learning_phase()], [model.layers[29].output])
	#print model.summary()
	#exit()
	return vgg_out

if __name__ == '__main__':
	vgg16_out = VGG_16('vgg16_weights.h5')
	# tokenizer = RegexpTokenizer(r'\w+')
	vgg16_nchannels, vgg16_nrows, vgg16_ncols = 3, 224, 224

	vid_ind = 0 # video_key_old = pickle.load(open('video_feat.pkl'))
	video_key = dict()
	video_val = dict()
	filelist = os.listdir('../data/y2t_list')
	for filename in filelist[1500:len(filelist)]:
		vid_ind += 1
		if '.txt' not in filename:
			continue
		ft = open('../data/y2t_list/'+filename).readlines()
		img_list = [np.transpose(misc.imresize(misc.imread(ft[i].strip()), (vgg16_nrows, vgg16_ncols, vgg16_nchannels)), (2,0,1)) for i in range(len(ft))]
		
		vgg_feat = (vgg16_out([img_list, 0])[0])
		vgg_feat = np.transpose(np.reshape(vgg_feat, (vgg_feat.shape[0], vgg_feat.shape[1], vgg_feat.shape[2] * vgg_feat.shape[3])), (0, 2, 1))
		# vgg_feat = np.sum(vgg_feat, 1)/ 196.0
		video_key[filename.replace('.txt', '')] = vgg_feat

		ft = h5py.File('../data/y2t_features_36/'+filename.replace('.txt','.h5'))
		scores = ft["boxes"]
		feats = ft["feats"]
		values = np.array([(scores[i].T.dot(feats[i])[0])/sum(scores[i]) for i in range(36)])
		video_val[filename.replace('.txt', '')] = values

		print vid_ind, filename, len(img_list), video_key[filename.replace('.txt', '')].shape, video_val[filename.replace('.txt', '')].shape

	with open('kv_att/video_keys_conv_att_3.pkl', 'w') as f:
		pickle.dump(video_key, f, protocol = 2)
	with open('kv_att/video_values_conv_att_3.pkl', 'w') as f:
		pickle.dump(video_val, f, protocol = 2)
		
	print ('vbsdjk')

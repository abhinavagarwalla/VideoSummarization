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

#-*- coding: utf-8 -*-
import math
import os
#import ipdb

# with tf.device('/cpu:0'):
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
    return vgg_out

class Caption_Generator():

	def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
		return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

	def init_bias(self, dim_out, name=None):
		return tf.Variable(tf.zeros([dim_out]), name=name)

	def __init__(self, n_words, dim_embed, dim_ctx, dim_hidden, n_lstm_steps, batch_size=200, ctx_shape=[196,512], bias_init_vector=None):
		self.n_words = n_words
		self.dim_embed = dim_embed
		self.dim_ctx = dim_ctx
		self.dim_hidden = dim_hidden
		self.ctx_shape = ctx_shape
		self.n_lstm_steps = n_lstm_steps
		self.batch_size = batch_size

		with tf.device("/cpu:0"):
			self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -1.0, 1.0), name='Wemb')

		self.init_hidden_W = self.init_weight(dim_ctx, dim_hidden, name='init_hidden_W')
		self.init_hidden_b = self.init_bias(dim_hidden, name='init_hidden_b')

		self.init_memory_W = self.init_weight(dim_ctx, dim_hidden, name='init_memory_W')
		self.init_memory_b = self.init_bias(dim_hidden, name='init_memory_b')

		self.lstm_W = self.init_weight(dim_embed, dim_hidden*4, name='lstm_W')
		self.lstm_U = self.init_weight(dim_hidden, dim_hidden*4, name='lstm_U')
		self.lstm_b = self.init_bias(dim_hidden*4, name='lstm_b')

		self.image_encode_W = self.init_weight(dim_ctx, dim_hidden*4, name='image_encode_W')

		self.image_att_W = self.init_weight(dim_ctx, dim_ctx, name='image_att_W')
		self.hidden_att_W = self.init_weight(dim_hidden, dim_ctx, name='hidden_att_W')
		self.pre_att_b = self.init_bias(dim_ctx, name='pre_att_b')

		self.att_W = self.init_weight(dim_ctx, 1, name='att_W')
		self.att_b = self.init_bias(1, name='att_b')

		self.decode_lstm_W = self.init_weight(dim_hidden, dim_embed, name='decode_lstm_W')
		self.decode_lstm_b = self.init_bias(dim_embed, name='decode_lstm_b')

		self.decode_word_W = self.init_weight(dim_embed, n_words, name='decode_word_W')

		if bias_init_vector is not None:
			self.decode_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='decode_word_b')
		else:
			self.decode_word_b = self.init_bias(n_words, name='decode_word_b')


	def get_initial_lstm(self, mean_context):
		initial_hidden = tf.nn.tanh(tf.matmul(mean_context, self.init_hidden_W) + self.init_hidden_b)
		initial_memory = tf.nn.tanh(tf.matmul(mean_context, self.init_memory_W) + self.init_memory_b)

		return initial_hidden, initial_memory

	def build_model(self):
		context = tf.placeholder("float32", [self.batch_size, self.ctx_shape[0], self.ctx_shape[1]])
		sentence = tf.placeholder("int32", [self.batch_size, self.n_lstm_steps])
		mask = tf.placeholder("float32", [self.batch_size, self.n_lstm_steps])

		h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))

		context_flat = tf.reshape(context, [-1, self.dim_ctx])
		context_encode = tf.matmul(context_flat, self.image_att_W) # (batch_size, 196, 512)
		context_encode = tf.reshape(context_encode, [-1, ctx_shape[0], ctx_shape[1]])

		loss = 0.0


		for ind in range(self.n_lstm_steps):

			if ind == 0:
				word_emb = tf.zeros([self.batch_size, self.dim_embed])
			else:
				tf.get_variable_scope().reuse_variables()
				with tf.device("/cpu:0"):
					word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,ind-1])

			x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b # (batch_size, hidden*4)

			labels = tf.expand_dims(sentence[:,ind], 1)
			indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
			concated = tf.concat(1, [indices, labels])
			onehot_labels = tf.sparse_to_dense( concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

			context_encode = context_encode + \
				 tf.expand_dims(tf.matmul(h, self.hidden_att_W), 1) + \
				 self.pre_att_b

			context_encode = tf.nn.tanh(context_encode)

			# context_encode: 3D -> flat required
			context_encode_flat = tf.reshape(context_encode, [-1, self.dim_ctx]) # (batch_size*196, 512)
			alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b # (batch_size*196, 1)
			alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]])
			alpha = tf.nn.softmax( alpha )

			weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)

			lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W)
			i, f, o, new_c = tf.split(1, 4, lstm_preactive)

			i = tf.nn.sigmoid(i)
			f = tf.nn.sigmoid(f)
			o = tf.nn.sigmoid(o)
			new_c = tf.nn.tanh(new_c)

			c = f * c + i * new_c
			h = o * tf.nn.tanh(new_c)

			logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
			logits = tf.nn.relu(logits)
			logits = tf.nn.dropout(logits, 0.5)

			logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
			cross_entropy = cross_entropy * mask[:,ind]

			current_loss = tf.reduce_sum(cross_entropy)
			loss = loss + current_loss

		loss = loss / tf.reduce_sum(mask)
		return loss, context, sentence, mask

# change in this function
	def build_generator(self, maxlen):
		context = tf.placeholder("float32", [None, self.ctx_shape[0], self.ctx_shape[1]])
		h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))

		context_flat = tf.reshape(context, [-1, self.dim_ctx])
		context_encode = tf.matmul(context_flat, self.image_att_W) # (batch_size, 196, 512)
		context_encode = tf.reshape(context_encode, [-1, ctx_shape[0], ctx_shape[1]])
		#context_encode = tf.matmul(tf.squeeze(context), self.image_att_W)
		generated_words = []
		logit_list = []
		alpha_list = []
		hidden_state_list = []
		word_emb = tf.zeros([tf.shape(context)[0], self.dim_embed])
		for ind in range(maxlen):
			x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b
			context_encode = context_encode + tf.expand_dims(tf.matmul(h, self.hidden_att_W), 1) + self.pre_att_b
			#context_encode = context_encode + tf.matmul(h, self.hidden_att_W) + self.pre_att_b
			context_encode = tf.nn.tanh(context_encode)
			context_encode_flat = tf.reshape(context_encode, [-1, self.dim_ctx])

			alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b
			alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]] )
			alpha = tf.nn.softmax(alpha)

			# alpha = tf.reshape(alpha, (ctx_shape[0], -1))
			# alpha_list.append(alpha)

			weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)
			#			weighted_context = tf.reduce_sum(tf.squeeze(context) * alpha, 0)
			#weighted_context = tf.expand_dims(weighted_context, 0)

			lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W)

			i, f, o, new_c = tf.split(1, 4, lstm_preactive)

			i = tf.nn.sigmoid(i)
			f = tf.nn.sigmoid(f)
			o = tf.nn.sigmoid(o)
			new_c = tf.nn.tanh(new_c)

			c = f*c + i*new_c
			h = o*tf.nn.tanh(new_c)
			hidden_state_list.append(h)


			logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
			logits = tf.nn.relu(logits)

			logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b

			max_prob_word = tf.argmax(logit_words, 1)

			with tf.device("/cpu:0"):
				word_emb = tf.nn.embedding_lookup(self.Wemb, max_prob_word)

			generated_words.append(max_prob_word)
			logit_list.append(logit_words)

		return context, generated_words, logit_list, alpha_list, hidden_state_list


def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # borrowed this function from NeuralTalk
	print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
	word_counts = {}
	nsents = 0
	for sent in sentence_iterator:
	  nsents += 1
	  for w in sent.lower().split(' '):
		word_counts[w] = word_counts.get(w, 0) + 1
	vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
	print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

	ixtoword = {}
	ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
	wordtoix = {}
	wordtoix['#START#'] = 0 # make first vector be the start token
	ix = 1
	for w in vocab:
	  wordtoix[w] = ix
	  ixtoword[ix] = w
	  ix += 1

	word_counts['.'] = nsents
	bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
	bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
	bias_init_vector = np.log(bias_init_vector)
	bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
	return wordtoix, ixtoword, bias_init_vector

n_epochs=1000
batch_size=120
dim_embed=256
dim_ctx=512
dim_hidden=256
ctx_shape=[196,512]
pretrained_model_path = None
#############################
annotation_path = 'annotations.pickle'
# feat_path = '../../show_attend_and_tell.tensorflow/data/feats.npy'
model_path = 'model-35'
#############################

def finish_parsing():
	global feat_path, model_path
	parser = argparse.ArgumentParser(description= "Forward pass an image_context and generate the sentence for it")
	parser.add_argument("--i",
						help="Path to feature.npy file")
	parser.add_argument("--m",
						help="Path to saved model")
	
	args = parser.parse_args()
	if args.i is not None:
		feat_path = os.path.abspath(args.i)
		print "Features located at %s" % feat_path
	if args.m is not None:
		model_path = os.path.abspath(args.m)
		print "Saved model at %s" % model_path


if __name__ == '__main__':
	annotation_data = pd.read_pickle(annotation_path)
	captions = annotation_data['caption'].values
	wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)
	n_words = len(wordtoix)

	print 'Starting session', n_words
	sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
	print 'interetdd'
	caption_generator = Caption_Generator(
			n_words=n_words,
			dim_embed=dim_embed,
			dim_ctx=dim_ctx,
			dim_hidden=dim_hidden,
			n_lstm_steps=100,
			batch_size=batch_size,
			ctx_shape=ctx_shape)
	context, generated_words, logit_list, alpha_list, hidden_state_list = caption_generator.build_generator(maxlen=100)
	saver = tf.train.Saver()
	saver.restore(sess, model_path)
	##################################################################

	vgg16_out = VGG_16('vgg16_weights.h5')
	# tokenizer = RegexpTokenizer(r'\w+')
	vgg16_nchannels, vgg16_nrows, vgg16_ncols = 3, 224, 224
	data = []
	raw_datapath = '../data/youtubeclips-dataset/'
	preprocessed_datapath = ''
	x_list, y_list = [], []

	vid_ind = 0 # video_key_old = pickle.load(open('video_feat.pkl'))
	video_key = dict()
	video_val = dict()
	for filename in os.listdir('../data/youtubeclips-dataset'):
		vid_ind += 1
		if '.avi' not in filename:
			continue
		video = VideoFileClip('../data/youtubeclips-dataset/' + filename)
		img_list = []
		for frame in video.iter_frames():
			img_list.append(np.transpose(misc.imresize(frame, (vgg16_nrows, vgg16_ncols, vgg16_nchannels)), (2, 0, 1)))

		num_frames = 52
		vgg_feat = (vgg16_out([img_list[:num_frames], 0])[0])
		vgg_feat = np.transpose(np.reshape(vgg_feat, (vgg_feat.shape[0], vgg_feat.shape[1], vgg_feat.shape[2] * vgg_feat.shape[3])), (0, 2, 1))
		video_key[filename.replace('.avi', '')] = vgg_feat
		
		generated_word_index = sess.run(generated_words, feed_dict={context:vgg_feat})
		generated_words_tmp = [ixtoword[x[0]] for x in generated_word_index]
		punctuation = np.argmax(np.array(generated_words_tmp) == '.')+1
		values = (sess.run([hidden_state_list[punctuation]], feed_dict={context:vgg_feat}))[0]
		video_val[filename.replace('.avi', '')] = values
		# print vgg_feat.shape, video_val[filename.replace('.avi', '')].shape, type(values)

		for j in range(1, ((len(img_list) - 1) / num_frames) + 1):
			vgg_feat = vgg16_out([img_list[num_frames * j: num_frames * (j + 1)], 0])[0]
			vgg_feat = np.transpose(np.reshape(vgg_feat, (vgg_feat.shape[0], vgg_feat.shape[1], vgg_feat.shape[2] * vgg_feat.shape[3])), (0, 2, 1))
			video_key[filename.replace('.avi', '')] = np.append(video_key[filename.replace('.avi', '')], vgg_feat, 0)
			generated_word_index = sess.run(generated_words, feed_dict={context:vgg_feat})
			generated_words_tmp = [ixtoword[x[0]] for x in generated_word_index]
			punctuation = np.argmax(np.array(generated_words_tmp) == '.')+1
			values = (sess.run([hidden_state_list[punctuation]], feed_dict={context:vgg_feat}))[0]

			video_val[filename.replace('.avi', '')] = np.append(video_val[filename.replace('.avi', '')], values, 0)
		
		video_key[filename.replace('.avi', '')] = ((np.sum(video_key[filename.replace('.avi', '')], 1)) / 196.0)
		print vid_ind, filename, len(img_list), video_key[filename.replace('.avi', '')].shape, video_val[filename.replace('.avi', '')].shape
		#vid_ind += 1

	with open('video_keys.pkl', 'w') as f:
		pickle.dump(video_key, f, protocol = 2)
	with open('video_values.pkl', 'w') as f:
		pickle.dump(video_val, f, protocol = 2)
		
	print ('vbsdjk')

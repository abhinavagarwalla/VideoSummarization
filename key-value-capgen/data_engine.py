import cPickle as pkl
import gzip
import os, socket, shutil
import sys, re
import time, pickle
from collections import OrderedDict
import numpy
import theano
import theano.tensor as T
import common

from multiprocessing import Process, Queue, Manager

hostname = socket.gethostname()
				
class Movie2Caption(object):
			
	def __init__(self, model_type, signature, video_feature,
				 mb_size_train, mb_size_test, maxlen, n_words,
				 n_frames=None, outof=None
				 ):
		self.signature = signature
		self.model_type = model_type
		self.video_feature = video_feature
		self.maxlen = maxlen
		self.n_words = n_words
		self.K = n_frames
		self.OutOf = outof
		self.ctx_dim = 196
		self.value_dim = 196

		self.mb_size_train = mb_size_train
		self.mb_size_test = mb_size_test
		self.non_pickable = []
		
		self.word_idict = pickle.load(open('../preprocessing/data/idx_to_word.pkl'))
		self.kf_train = range(100)
		self.kf_valid = range(100, 110)
		self.kf_test = range(110, 120)
		# self.load_data()
		
	def _filter_googlenet(self, vidID):
		feat = self.FEAT[vidID]
		feat = self.get_sub_frames(feat)
		return feat
	
	def get_video_features(self, vidID):
		if self.video_feature == 'googlenet':
			y = self._filter_googlenet(vidID)
		else:
			raise NotImplementedError()
		return y

	def pad_frames(self, frames, limit, jpegs):
		# pad frames with 0, compatible with both conv and fully connected layers
		last_frame = frames[-1]
		if jpegs:
			frames_padded = frames + [last_frame]*(limit-len(frames))
		else:
			padding = numpy.asarray([last_frame * 0.]*(limit-len(frames)))
			frames_padded = numpy.concatenate([frames, padding], axis=0)
		return frames_padded
	
	def extract_frames_equally_spaced(self, frames, how_many):
		# chunk frames into 'how_many' segments and use the first frame
		# from each segment
		n_frames = len(frames)
		splits = numpy.array_split(range(n_frames), self.K)
		idx_taken = [s[0] for s in splits]
		sub_frames = frames[idx_taken]
		return sub_frames
	
	def add_end_of_video_frame(self, frames):
		if len(frames.shape) == 4:
			# feat from conv layer
			_,a,b,c = frames.shape
			eos = numpy.zeros((1,a,b,c),dtype='float32') - 1.
		elif len(frames.shape) == 2:
			# feat from full connected layer
			_,b = frames.shape
			eos = numpy.zeros((1,b),dtype='float32') - 1.
		else:
			import pdb; pdb.set_trace()
			raise NotImplementedError()
		frames = numpy.concatenate([frames, eos], axis=0)
		return frames
	
	def get_sub_frames(self, frames, jpegs=False):
		# from all frames, take K of them, then add end of video frame
		# jpegs: to be compatible with visualizations
		if self.OutOf:
			raise NotImplementedError('OutOf has to be None')
			frames_ = frames[:self.OutOf]
			if len(frames_) < self.OutOf:
				frames_ = self.pad_frames(frames_, self.OutOf, jpegs)
		else:
			if len(frames) < self.K:
				#frames_ = self.add_end_of_video_frame(frames)
				frames_ = self.pad_frames(frames, self.K, jpegs)
			else:

				frames_ = self.extract_frames_equally_spaced(frames, self.K)
				#frames_ = self.add_end_of_video_frame(frames_)
		if jpegs:
			frames_ = numpy.asarray(frames_)
		return frames_

	def prepare_data_for_blue(self, whichset):
		# assume one-to-one mapping between ids and features
		feats = []
		feats_mask = []
		if whichset == 'valid':
			ids = self.valid_ids
		elif whichset == 'test':
			ids = self.test_ids
		elif whichset == 'train':
			ids = self.train_ids
		for i, vidID in enumerate(ids):
			feat = self.get_video_features(vidID)
			feats.append(feat)
			feat_mask = self.get_ctx_mask(feat)
			feats_mask.append(feat_mask)
		return feats, feats_mask
	
	def get_ctx_mask(self, ctx):
		if ctx.ndim == 3:
			rval = (ctx[:,:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
		elif ctx.ndim == 2:
			rval = (ctx[:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
		elif ctx.ndim == 5 or ctx.ndim == 4:
			# assert self.video_feature == 'oxfordnet_conv3_512'
			# in case of oxfordnet features
			# (m, 26, 512, 14, 14)
			rval = (ctx.sum(-1).sum(-1).sum(-1) != 0).astype('int32').astype('float32')
		else:
			import pdb; pdb.set_trace()
			raise NotImplementedError()
		
		return rval
	
		
	def load_data(self):
		if self.signature == 'youtube2text':
			print 'loading youtube2text %s features'%self.video_feature
			dataset_path = common.get_rab_dataset_base_path()+'youtube2text_iccv15/'
			self.train = common.load_pkl(dataset_path + 'train.pkl')
			self.valid = common.load_pkl(dataset_path + 'valid.pkl')
			self.test = common.load_pkl(dataset_path + 'test.pkl')
			self.CAP = common.load_pkl(dataset_path + 'CAP.pkl')
			self.FEAT = common.load_pkl(dataset_path + 'FEAT_key_vidID_value_features.pkl')
			self.train_ids = ['vid%s'%i for i in range(1,1201)]
			self.valid_ids = ['vid%s'%i for i in range(1201,1301)]
			self.test_ids = ['vid%s'%i for i in range(1301,1971)]
		else:
			raise NotImplementedError()
				
		self.worddict = common.load_pkl(dataset_path + 'worddict.pkl')
		self.word_idict = dict()
		# wordict start with index 2
		for kk, vv in self.worddict.iteritems():
			self.word_idict[vv] = kk
		self.word_idict[0] = '<eos>'
		self.word_idict[1] = 'UNK'
		
		if self.video_feature == 'googlenet':
			self.ctx_dim = 1024
		else:
			raise NotImplementedError()
		self.kf_train = common.generate_minibatch_idx(
			len(self.train), self.mb_size_train)
		self.kf_valid = common.generate_minibatch_idx(
			len(self.valid), self.mb_size_test)
		self.kf_test = common.generate_minibatch_idx(
			len(self.test), self.mb_size_test)
		
def prepare_data(engine, index):
	seqs = []
	feat_list = []
	
	# file_num = index / 4
	# file_set = index % 4
	file_num = index
	path = '../preprocessing/data/'
	
	x_file = 'X_' + str(file_num) + '.pkl'
	y_file = 'Y_' + str(file_num) + '.pkl'

	feat_list = pickle.load(open(path + x_file))[:16] #[128 * file_set:128 * (file_set + 1)]
	seq_list = pickle.load(open(path + y_file))[:16] #[128 * file_set:128 * (file_set + 1)]
	
	new_feat_list = []
	# print (feat_list[0].shape, len(feat_list))
	# print (len(seq_list), seq_list[0])
	for i in range(len(feat_list)):
		new_feat_list.append(engine.get_sub_frames(feat_list[i]))

	feat_list = new_feat_list

	lengths = [len(s) for s in seq_list]
	# if engine.maxlen != None:
	#     new_seqs = []
	#     new_feat_list = []
	#     new_lengths = []
	#     new_caps = []
	#     for l, s, y, c in zip(lengths, seqs, feat_list, IDs):
	#         # sequences that have length >= maxlen will be thrown away 
	#         if l < engine.maxlen:
	#             new_seqs.append(s)
	#             new_feat_list.append(y)
	#             new_lengths.append(l)
	#             new_caps.append(c)
	#     lengths = new_lengths
	#     feat_list = new_feat_list
	#     seqs = new_seqs
	#     if len(lengths) < 1:
	#         return None, None, None, None
	
	y = numpy.asarray(feat_list)
	y = y / 512.0
	y = numpy.reshape(y, (y.shape[0], y.shape[1], y.shape[2] * y.shape[3]))
	y_mask = engine.get_ctx_mask(y)
	n_samples = len(seq_list)
	maxlen = numpy.max(lengths)+1

	#print (maxlen, 'maxlen is ')
	x = numpy.zeros((maxlen, n_samples)).astype('int64')
	x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
	for i, s in enumerate(seq_list):
		x[:lengths[i],i] = s
		x_mask[:lengths[i]+1,i] = 1.
	
	#print (x.shape, x_mask.shape, y.shape, y_mask.shape)
	return x, x_mask, y, y_mask, y, y_mask # passing the same thing as key and value respectively
	
def test_data_engine():
	from sklearn.cross_validation import KFold
	video_feature = 'googlenet' 
	out_of = None
	maxlen = 100
	mb_size_train = 64
	mb_size_test = 128
	maxlen = 50
	n_words = 30000 # 25770 
	signature = 'youtube2text' #'youtube2text'
	engine = Movie2Caption('attention', signature, video_feature,
						   mb_size_train, mb_size_test, maxlen,
						   n_words,
						   n_frames=26,
						   outof=out_of)
	i = 0
	t = time.time()
	for idx in engine.kf_train:
		t0 = time.time()
		i += 1
		ids = [engine.train[index] for index in idx]
		x, mask, ctx, ctx_mask = prepare_data(engine, ids)
		print 'seen %d minibatches, used time %.2f '%(i,time.time()-t0)
		if i == 10:
			break
			
	print 'used time %.2f'%(time.time()-t)
if __name__ == '__main__':
	test_data_engine()



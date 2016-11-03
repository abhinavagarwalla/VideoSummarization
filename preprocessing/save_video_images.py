import multiprocessing
import os
from moviepy.editor import *
from glob import glob
import numpy as np
import math

def save_images(filename):
	if '.avi' not in filename:
		return
	video = VideoFileClip('../data/youtubeclips-dataset/' + filename)
	video.to_images_sequence('../data/y2t_images/'+filename.split('.')[0]+'/%s.jpeg')

def make_dirs(filename):
	if '.avi' not in filename:
		return
	os.mkdir('../data/y2t_images/' + filename.split('.')[0])

def make_list(filename):
	if '.avi' not in filename:
		return
	fsplit = filename.split('.')[0]
	#flist = os.listdir('../data/y2t_images/'+fsplit)
	flist = glob('/home/abhinav/Desktop/VideoSummarization/data/y2t_images/'+fsplit+'/*')
	flist = sorted(flist, key = lambda x: float(x.split('.')[0].split('/')[-1]))
	print len(flist), len(flist)/28
	#ik =  np.arange(0, len(flist)-len(flist)/28, len(flist)/28.)
	#print len(ik), ik
	k = [flist[int(i)] for i in np.arange(0,len(flist)-1, len(flist)/28.)]
	print len(k)
	if len(k)!=28:
		exit()
	fw = open('../data/y2t_list/'+fsplit+'.txt', 'w')
	fw.write("\n".join(k))
	fw.write("\n")
	fw.close()

if __name__ == '__main__':
	#filelist = os.listdir('../data/youtubeclips-dataset')
	filelist = glob('/home/abhinav/Desktop/VideoSummarization/data/y2t_list/*')
	for i in range(1800,1801,200):
		fw = open('../data/input_list_'+str(i)+'.txt', 'w')
		fw.write("\n".join(filelist[i:min(len(filelist),i+200)]))
		fw.write("\n")
		fw.close()
	# print filelist
	#make_list('vid998.avi')
	#exit()
	#multiprocessing.Pool().map(save_images, flist)
	#multiprocessing.Pool().map(make_list, filelist)
	#for i in filelist:
	#	print i
	#	make_list(i)
	print "All done"

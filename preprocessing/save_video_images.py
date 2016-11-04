import multiprocessing
import os
from moviepy.editor import *
from glob import glob

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
	flist = glob('/home/bt0/12CS10045/VideoSummarization/data/y2t_images/'+fsplit+'/*')
	flist = sorted(flist, key = lambda x: float(x.split('.')[0].split('/')[-1]))
	fw = open('../data/y2t_list/'+fsplit+'.txt', 'w')
	fw.write("\n".join(flist))
	fw.write("\n")
	fw.close()

if __name__ == '__main__':
	#flist = os.listdir('../data/youtubeclips-dataset')
	filelist = glob('/home/bt0/12CS10045/VideoSummarization/data/y2t_list/*')
	for i in range(0,1950, 50):
		fw = open('../data/input_list_'+str(i)+'.txt', 'w')
		fw.write("\n".join(filelist))
		fw.write("\n")
		fw.close()
	# print filelist
	#make_list('vid100.avi')
	#multiprocessing.Pool().map(make_list, flist)
	#multiprocessing.Pool().map(make_list, filelist)
	

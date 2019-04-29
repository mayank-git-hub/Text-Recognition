import json
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pickle
from shutil import copyfile
import re


class MetaSynth():

	def __init__(self, config):

		self.config = config
		self.seed()

	def seed(self):

		#Causes randomness to start from same initial point, hence deterministic
		#This is important because each time we initialize a new model, the reults are identical if we perform same experiments\

		np.random.seed(self.config['seed'])

	def split(self, list_all):

		split_r = float(self.config['metadata']['SYNTH']['split_type'])#PUT IN YAML FILE!!!!!!!
		#defines ratio of training to total images

		# Creating a list of filenames for which annotation is available

		idx = np.arange(len(list_all))#list with array from 0 to len(list_all)-1
		np.random.shuffle(idx)#randomly shuffles list(deterministic)

		# /home/mayank/Desktop/GitRepos/Text/Segmentation/Dataset/labels_text/meta
		# 
		if os.path.exists(self.config['metadata']['SYNTH']['meta']+'/train_files_'+str(split_r)+'.txt'):
			os.remove(self.config['metadata']['SYNTH']['meta']+'/train_files_'+str(split_r)+'.txt')#remove file containing file names of training images

		f = open(self.config['metadata']['SYNTH']['meta']+'/train_files_'+str(split_r)+'.txt', 'w')#opens file in write mode

		#forms two arrays splitting all possible image paths into training and testing

		train = np.array(list_all)[idx[:int(split_r*len(list_all))]]
		val = np.array(list_all)[idx[int(split_r*len(list_all)):]]
		for i in train:
			f.write(i+'\n') #writes path name into .txt file for training

		#similar process done for test files

		if os.path.exists(self.config['metadata']['SYNTH']['meta']+'/test_files_'+str(split_r)+'.txt'):
			os.remove(self.config['metadata']['SYNTH']['meta']+'/test_files_'+str(split_r)+'.txt')

		f = open(self.config['metadata']['SYNTH']['meta']+'/test_files_'+str(split_r)+'.txt', 'w')

		for i in val:
			f.write(i+'\n')

		return train

	def calc_average(self, train):

		image_average = [np.zeros(3), 0] #np.zeros(3) gives array of floats [0.0,0.0,0.0]
		image_std = [np.zeros(3), 0]

		random_images = np.random.choice(train, min(1000, len(train)), replace=False)
		#Array of random choices from train, with length at max 1000, wihtout replacement

		for i in random_images:

			ext = [ext_i for ext_i in ['.png', '.jpg'] if os.path.exists(self.config['metadata']['SYNTH']['image']+'/'+i+ext_i)]
			if len(ext) == 0:
				print('Error: File not found', i)

			ext = ext[0]

			image = plt.imread(self.config['metadata']['SYNTH']['image']+'/'+i+ext)
			if len(image.shape) == 2: #if greyscale, continue
				continue
			image_average[0] += np.sum(image, axis=(0, 1))
			image_average[1] += image.shape[0]*image.shape[1]

		image_average[0] = image_average[0]/image_average[1]

		#average of all the images in random images calculated

		for i in random_images:

			ext = [ext_i for ext_i in ['.png', '.jpg'] if os.path.exists(self.config['metadata']['SYNTH']['image']+'/'+i+ext_i)]
			if len(ext) == 0:
				print('Error: File not found', i)

			ext = ext[0]

			image = plt.imread(self.config['metadata']['SYNTH']['image']+'/'+i+'.png')
			if len(image.shape) == 2:
				continue
			image_std[0] += np.sum(np.square(image - image_average[0]), axis=(0, 1))
			image_std[1] += image.shape[0]*image.shape[1]

		image_std[0] = image_std[0]/image_std[1]

		#std deviation of all the images in random images calculated

		with open(self.config['metadata']['SYNTH']['meta']+'/normalisation_'+str(split_r)+'.pkl', 'wb') as f:
			pickle.dump({'average': image_average, 'std': image_std}, f)

	def create_annot(self):

		#This function loads annotations on images from a matlab file
		#Number of annotations CAN BE ADDED to yaml file

		import scipy.io

		mat = scipy.io.loadmat(self.config['metadata']['SYNTH']['dir']+'/UnClean/gt.mat')
		list_all = []

		for i in range(858750):

			filename = mat['imnames'][0, i][0].split('/')[0]+'_'+'/'.join(mat['imnames'][0, i][0].split('/')[1:])+'.pkl'

			if len(mat['wordBB'][0, i].shape) == 2:
				annots = mat['wordBB'][0, i][:, :, None].transpose(2, 1, 0)
			else:
				annots = mat['wordBB'][0, i].transpose(2, 1, 0)

			annots = annots.reshape(annots.shape[0], annots.shape[1], 1, 2).astype(np.int32)
			text_annots = (' '.join((' '.join(mat['txt'][0, i])).split('\n'))).split()

			with open(self.config['metadata']['SYNTH']['label']+'/'+filename, 'wb') as f:

				pickle.dump([annots, text_annots], f)

			file = '.'.join(mat['imnames'][0, i][0].split('.')[:-1])
			file = file.split('/')[0]+'_'+'/'.join(file.split('/')[1:])+'.jpg'

			list_all.append(file)

		train = self.split(list_all) #Splits it into train and test
		#Because of self.seed()the order is deterministic

		if self.config['metadata']['SYNTH']['cal_avg']:
			self.calc_average(train)

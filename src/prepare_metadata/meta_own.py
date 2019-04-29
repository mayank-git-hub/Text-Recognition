import json
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pickle
from shutil import copyfile
import re

class MetaOwn():

	def __init__(self, config):

		self.config = config
		self.seed()
		self.split_type = self.config['metadata']['OWN']['split_type']
		
	def seed(self):

		np.random.seed(self.config['seed'])

	def create_annot(self):

		all_list = os.listdir(self.config['metadata']['OWN']['image'])
		train = self.split_ratio(all_list, float(self.split_type))
		if self.config['metadata']['OWN']['cal_avg']:
			self.calc_average(train)

	def split_ratio(self, all_list, train_ratio):

		#Forms the training and testing image paths, by splitting the dataset, deterministic due to self.seed()

		idx = np.arange(len(all_list))
		np.random.shuffle(idx)

		# /home/mayank/Desktop/GitRepos/Text/Segmentation/Dataset/labels_text/meta

		if os.path.exists(self.config['metadata']['OWN']['meta']+'/train_files_'+str(train_ratio)+'.txt'):
			os.remove(self.config['metadata']['OWN']['meta']+'/train_files_'+str(train_ratio)+'.txt')

		f = open(self.config['metadata']['OWN']['meta']+'/train_files_'+str(train_ratio)+'.txt', 'w')

		train = np.array(all_list)[idx[:int(train_ratio*len(all_list))]]
		val = np.array(all_list)[idx[int(train_ratio*len(all_list)):]]

		for i in train:
			f.write(i+'\n')

		if os.path.exists(self.config['metadata']['OWN']['meta']+'/test_files_'+str(train_ratio)+'.txt'):
			os.remove(self.config['metadata']['OWN']['meta']+'/test_files_'+str(train_ratio)+'.txt')

		f = open(self.config['metadata']['OWN']['meta']+'/test_files_'+str(train_ratio)+'.txt', 'w')

		for i in val:
			f.write(i+'\n')

	def calc_average(self, train):

		image_average = [np.zeros(3), 0]
		image_std = [np.zeros(3), 0]

		random_images = np.random.choice(train, min(1000, len(train)), replace=False)

		for i in random_images:

			image = plt.imread(self.config['metadata']['OWN']['image']+'/'+i)
			if len(image.shape) == 2:
				continue
			image_average[0] += np.sum(image, axis=(0, 1))
			image_average[1] += image.shape[0]*image.shape[1]

		image_average[0] = image_average[0]/image_average[1]

		for i in random_images:

			image = plt.imread(self.config['metadata']['OWN']['image']+'/'+i)
			if len(image.shape) == 2:
				continue
			image_std[0] += np.sum(np.square(image - image_average[0]), axis=(0, 1))
			image_std[1] += image.shape[0]*image.shape[1]

		image_std[0] = image_std[0]/image_std[1]

		with open(self.config['metadata']['OWN']['meta']+'/normalisation_'+str(train_ratio)+'.pkl', 'wb') as f:
			pickle.dump({'average': image_average, 'std': image_std}, f)
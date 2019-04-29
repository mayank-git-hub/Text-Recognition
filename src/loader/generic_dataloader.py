import torch.utils.data as data
import torch
import pickle

import os
import numpy as np
import time  #

from PIL import Image

import cv2
import matplotlib.pyplot as plt
from ..helper.logger import Logger
import json
from ..helper.profiler import Profiler

log = Logger()


class own_DataLoader():

	def __init__(self, config, Type, dataloader_type='scale_two', profiler=None, **kwargs):

		# constructor called

		self.config = config
		self.seed()

		if Type != 'test_one':
			self.datasets = sorted(config['dataset_' + Type])
			self.cache_name = self.config['cache_path'] + '/' + '_'.join(self.datasets) + '_' + Type + '_' + \
			                  self.config['project'] + '.pkl'

		if profiler:
			self.profiler = profiler

		if Type != 'test_one':
			self.cache_name = self.config['cache_path'] + '/' + '_'.join(self.datasets) + '_' + Type + '_' + \
			                  self.config['project'] + '.pkl'
			self.datasets_attr = {}

			for dataset in self.datasets:
				self.datasets_attr[dataset] = {}

				self.datasets_attr[dataset]['split_type'] = self.config['metadata'][dataset]['split_type']
				self.datasets_attr[dataset]['root'] = config['metadata'][dataset]['dir']
				self.datasets_attr[dataset]['image_root'] = config['metadata'][dataset]['image']
				self.datasets_attr[dataset]['label_root'] = config['metadata'][dataset]['label']
				self.datasets_attr[dataset]['meta'] = config['metadata'][dataset]['meta']
				self.datasets_attr[dataset]['contour_area_thresh_min'] = config['metadata'][dataset][
					'contour_area_thresh_min']
				self.datasets_attr[dataset]['contour_length_thresh_min'] = config['metadata'][dataset][
					'contour_length_thresh_min']

		self.Type = Type
		self.dataloader_type = dataloader_type

		if self.dataloader_type == 'square':
			if 'image_size' in config[Type]:
				self.image_size = config[Type]['image_size']
			else:
				self.image_size = config['image_size']

		self.IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif']
		self.transform = kwargs['transform']
		self.target_transform = kwargs['target_transform']
		self.batchsize = config[Type]['batch_size']

		if not config[Type]['loader']['flag']:
			self.loader = self.pil_loader
		else:
			self.loader = kwargs['loader']

		self.char_to_encoding = {}
		self.encoding_to_char = {}

		all_char = False

		if not all_char:
			all_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
			                  'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
			                  '.', '/', '>', '-', '=', '<']
		# all_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']#, ':', '<', '>', '.', '?', '/'
		else:
			all_characters = ['!']
			all_characters += ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
			                   'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
			all_characters += ['!', 'é', 'É', ' ', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
			                   ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '´', '{', '|', '}',
			                   '~']
			all_characters += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

		self.abc = ''.join(all_characters)
		self.all_characters = all_characters

		print('Dictionary: ', self.abc)
		for no, c in enumerate(self.abc):
			self.char_to_encoding[c] = no + 1
			self.encoding_to_char[no + 1] = c

		# These two dictionaries, corresponding element is 1, rest are 0's

		if self.Type != 'test_one':
			profiler(self.get_all_names_refresh, attr=('get_all_names_refresh_' + self.Type))

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def _update_config(self, config):

		for dataset in self.datasets:

			if 'range' in self.datasets_attr[dataset]:
				self.datasets_attr[dataset] = {'range': self.datasets_attr[dataset]['range']}
			else:
				self.datasets_attr[dataset] = {}

			self.datasets_attr[dataset]['split_type'] = self.config['metadata'][dataset]['split_type']
			self.datasets_attr[dataset]['root'] = config['metadata'][dataset]['dir']
			self.datasets_attr[dataset]['image_root'] = config['metadata'][dataset]['image']
			self.datasets_attr[dataset]['label_root'] = config['metadata'][dataset]['label']
			self.datasets_attr[dataset]['meta'] = config['metadata'][dataset]['meta']

		if self.dataloader_type == 'square':
			if 'image_size' in config[self.Type]:
				self.image_size = config[self.Type]['image_size']
			else:
				self.image_size = config['image_size']

		self.batchsize = config[self.Type]['batch_size']

	def get_all_names_refresh(self):

		if self.Type == 'train':

			if self.config['cache'] and not os.path.exists(self.cache_name):

				self.process()

				with open(self.cache_name, 'wb') as f:
					pickle.dump({
						'images': self.images,
						'datasets_attr': self.datasets_attr,
						'annots': self.annots,
						'texts': self.texts,
						'remove_annots': self.remove_annots,
						'prob_sample': self.prob_sample,
					}, f)
				print("Successfully loaded and dumped", self.Type, 'cache')

			elif self.config['cache'] and os.path.exists(self.cache_name):

				with open(self.cache_name, 'rb') as f:

					dict_ = pickle.load(f)

				self.images = dict_['images']
				self.texts = dict_['texts']
				self.datasets_attr = dict_['datasets_attr']
				self.annots = dict_['annots']
				self.remove_annots = dict_['remove_annots']
				self.prob_sample = dict_['prob_sample']

				print("Successfully loaded", self.Type, 'cache')
				print('Total # images', len(self.images))

			else:

				self.process()
		else:

			self.process()

		if self.Type == 'test':
			self.start = 0

		self.weight_dataset = {'SYNTH': 1, 'COCO': 4, 'OWN': 2, 'ART': 0.5, 'IC13': 1, 'IC15': 10}
		self.prob_sample = []
		prob_sample_dataset = []

		for dataset_name in sorted(self.datasets_attr.keys()):
			data_dict = self.datasets_attr[dataset_name]
			current = data_dict['range'][1] - data_dict['range'][0]
			self.prob_sample += (np.ones([current]) * self.weight_dataset[dataset_name]).tolist()
			prob_sample_dataset.append(
				(data_dict['range'][1] - data_dict['range'][0]) * self.weight_dataset[dataset_name])

		self.prob_sample = np.array(self.prob_sample)
		self.prob_sample = self.prob_sample / self.prob_sample.sum()
		prob_sample_dataset = np.array(prob_sample_dataset)
		prob_sample_dataset = prob_sample_dataset / prob_sample_dataset.sum()

		for no, dataset_name in enumerate(sorted(self.datasets_attr.keys())):
			print('Probability of sampling: ', dataset_name, '= ', prob_sample_dataset[no])

	# This normalises std dev and mean to (0,1), then converted to float tensors
	# Lists for annotations added and removed formed
	# One hot encoding and reverse mapping of all possible characters formed in dictionary form

	def pil_loader(self, path):

		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

	def random_channel_shift(self, image):

		# Image is a numpy array with 3 channels

		channels = np.array([0, 1, 2])
		np.random.shuffle(channels)
		return image[:, :, channels]

	def process(self):

		self.weight_dataset = {'SYNTH': 1, 'COCO': 4, 'OWN': 1, 'ART': 0.5, 'IC13': 10, 'IC15': 10}

		images = {}
		total_images = 0
		self.prob_sample = []

		for dataset_name in sorted(self.datasets_attr.keys()):

			# if dataset_name == 'OWN':
			# 	continue

			data_dict = self.datasets_attr[dataset_name]

			current = 0
			split_type = data_dict['split_type']
			meta = data_dict['meta']
			image_root = data_dict['image_root']
			images[dataset_name] = []

			if split_type == 'original':
				f_path = meta + '/' + self.Type + '_files_original.txt'
			else:
				f_path = meta + '/' + self.Type + '_files_' + split_type + '.txt'

			with open(f_path, 'r') as f:

				for i in f:
					i = i[:-1]
					current += 1
					if os.path.exists(image_root + '/' + i):
						images[dataset_name].append(i)

					else:
						current -= 1
						print('Error: File not found', i, self.Type, i)

			data_dict['range'] = (total_images, total_images + current)
			total_images += current
			self.prob_sample += (np.ones([current]) * self.weight_dataset[dataset_name]).tolist()

		self.prob_sample = np.array(self.prob_sample)
		self.prob_sample = self.prob_sample / self.prob_sample.sum()

		self.annots = [[] for i in range(total_images)]  # list of list initalised to empty
		self.remove_annots = [[] for i in range(total_images)]
		self.texts = [[] for i in range(total_images)]

		annot_till_now = 0

		for dataset_name in sorted(images.keys()):

			# print("In dataset:",dataset_name)
			# initval = [0]*len(self.all_characters)
			# dataset_distribution = dict(zip(self.all_characters, initval))
			# print(dataset_distribution)

			images_list = images[dataset_name]

			label_root = self.datasets_attr[dataset_name]['label_root']
			thresh_area = self.datasets_attr[dataset_name]['contour_area_thresh_min']
			thresh_length = self.datasets_attr[dataset_name]['contour_length_thresh_min']

			for no, i in enumerate(images_list):

				# if no%1000 == 0:
				# 	print(dataset_distribution)
				# print(no)

				f = open(label_root + '/' + i + '.pkl', 'rb')

				annot_with_text = pickle.load(f)

				annot, all_text = annot_with_text
				# print(annot_with_text)
				# for t in all_text:
				# 	if ':' in t or '/' in t or '<' in t:
				# 		print(t, 'Done')

				cleaned_annot = []
				removed_annot = []
				text_annot = []

				for j, text in zip(annot, all_text):

					if text == None:
						text = ''
					if len(text) > 0:
						if ord(text[0]) > 128:
							# print('skipping', text)
							continue

					j[j <= 0] = 0
					text = text.lower()
					j = j.astype(np.int32)
					original = cv2.contourArea(j)
					if original == 0:
						# print(text)
						continue

					box = cv2.boxPoints(cv2.minAreaRect(j)).astype(np.int64).reshape(
						[4, 1, 2])  # min area rect applied to form bounding box

					new_text = []
					to_app = True
					for text_i in text:
						if text_i in self.char_to_encoding:
							new_text.append(self.char_to_encoding[text_i])
						else:
							# print("Skipping",text_i)
							new_text = []
							to_app = False
							break
					# if to_app:
					# 	for text_i in text:
					# 		dataset_distribution[text_i] += 1
					# 		if text_i in [':','/','<']:
					# 			print(text)

					text_annot.append(new_text)
					cleaned_annot.append(
						box)  # if threshold for area and lengths not satisfied, add to remove_annot, else to cleaned_annot

				self.annots[no + annot_till_now] = np.array(cleaned_annot)
				self.remove_annots[no + annot_till_now] = np.array(removed_annot)
				self.texts[no + annot_till_now] = text_annot
				# print("Adding")
				# if self.Type == 'train':
				# 	for t in text_annot:
				# 		# print(t)
				# 		a = [self.encoding_to_char[c] for c in t]
				# 		if ':' in a or '/' in a:
				# print('Adding')

				# if '.'.join(i.split('.')[:-1]) == 'passport-5':
				# 	print(annot_with_text[0].shape, len(annot_with_text[1]), self.annots[no+annot_till_now].shape)
				# 	exit(0)

				f.close()

			self.datasets_attr[dataset_name]['length'] = len(images_list)
			annot_till_now += len(images_list)

		self.images = []
		for name in sorted(images.keys()):
			l = images[name]
			self.images += l

	def _get_link(self, edge_contour, link, dummy_image):

		all_i = edge_contour[:, 1]
		all_j = edge_contour[:, 0]

		all_i_0 = all_i > 0
		all_j_0 = all_j > 0

		all_j_less = all_j < dummy_image.shape[1] - 1
		all_i_less = all_i < dummy_image.shape[0] - 1

		minus_all_i_thresh = all_i.copy()
		minus_all_i_thresh[np.logical_not(all_i_0)] = 1

		minus_all_j_thresh = all_j.copy()
		minus_all_j_thresh[np.logical_not(all_j_0)] = 1

		plus_all_i_thresh = all_i.copy()
		plus_all_i_thresh[np.logical_not(all_i_less)] = dummy_image.shape[0] - 2

		plus_all_j_thresh = all_j.copy()
		plus_all_j_thresh[np.logical_not(all_j_less)] = dummy_image.shape[1] - 2

		link[all_i, all_j, :] = 0

		temp = np.where(np.logical_and(np.logical_and(all_i_0, all_j_0),
		                               dummy_image[minus_all_i_thresh - 1, minus_all_j_thresh - 1] == 1))[0]
		link[all_i[temp], all_j[temp], 0] = 1

		temp = np.where(np.logical_and(all_i_0, dummy_image[minus_all_i_thresh - 1, all_j] == 1))[0]
		link[all_i[temp], all_j[temp], 1] = 1

		temp = np.where(np.logical_and(np.logical_and(all_i_0, all_j_less),
		                               dummy_image[minus_all_i_thresh - 1, plus_all_j_thresh + 1] == 1))[0]
		link[all_i[temp], all_j[temp], 2] = 1

		temp = np.where(np.logical_and(all_j_less, dummy_image[all_i, plus_all_j_thresh + 1] == 1))[0]
		link[all_i[temp], all_j[temp], 3] = 1

		temp = np.where(np.logical_and(np.logical_and(all_i_less, all_j_less),
		                               dummy_image[plus_all_i_thresh + 1, plus_all_j_thresh + 1] == 1))[0]
		link[all_i[temp], all_j[temp], 4] = 1

		temp = np.where(np.logical_and(all_i_less, dummy_image[plus_all_i_thresh + 1, all_j] == 1))[0]
		link[all_i[temp], all_j[temp], 5] = 1

		temp = np.where(np.logical_and(np.logical_and(all_i_less, all_j_0),
		                               dummy_image[plus_all_i_thresh + 1, minus_all_j_thresh - 1] == 1))[0]
		link[all_i[temp], all_j[temp], 6] = 1

		temp = np.where(np.logical_and(all_j_0, dummy_image[all_i, minus_all_j_thresh - 1] == 1))[0]
		link[all_i[temp], all_j[temp], 7] = 1

		return link

	def get_link(self, contours, resized_image_shape, ret_all=True):

		target = np.zeros([resized_image_shape[0], resized_image_shape[1]]).astype(np.uint8)

		if self.config['link'] and ret_all:
			link = np.zeros([resized_image_shape[0], resized_image_shape[1],
			                 8])  # 8 channels, up down left right, left up, left down, right up, right down

		if self.config['weight_bbox'] and ret_all:
			weight = np.zeros([resized_image_shape[0], resized_image_shape[1]])

		"""
		Up-left, Up, Up-right, right, Down-right, Down, Down-left, left
		"""

		for no, contour in enumerate(contours):

			sum_ = cv2.contourArea(contour)
			if sum_ == 0:
				log.info('Found an image which has zero area contour', contour)
				continue

			dummy_image = np.zeros([resized_image_shape[0], resized_image_shape[1]]).astype(np.uint8)
			cv2.drawContours(dummy_image, [contour], -1, 1, cv2.FILLED)

			# getting minimum square bbox
			min_x, min_y = np.min(contour.reshape([contour.shape[0], 2]), axis=0) - 1
			max_x, max_y = np.max(contour.reshape([contour.shape[0], 2]), axis=0) + 1

			min_x = max(0, min_x)
			min_y = max(0, min_y)
			max_x = min(resized_image_shape[1], max_x)
			max_y = min(resized_image_shape[0], max_y)

			intersection = np.logical_and(dummy_image[min_y:max_y, min_x:max_x],
			                              target[min_y:max_y, min_x:max_x]).astype(np.uint8)

			# dummy image fills those regions bounded by contours with ones
			# logical and taken with image so that intersection only contains regions in original image that are bounded by contours

			dummy_image[min_y:max_y, min_x:max_x] -= intersection
			target[min_y:max_y, min_x:max_x] = target[min_y:max_y, min_x:max_x] + dummy_image[min_y:max_y,
			                                                                      min_x:max_x] - intersection

			if self.config['weight_bbox'] and ret_all:
				weight[min_y:max_y, min_x:max_x] = weight[min_y:max_y, min_x:max_x] + dummy_image[min_y:max_y,
				                                                                      min_x:max_x] / sum_

			# Weight initialised to value in (0,1) inversely proportional to the area in the bounding box

			if self.config['link'] and ret_all:

				link[min_y:max_y, min_x:max_x] = link[min_y:max_y, min_x:max_x] * target[min_y:max_y, min_x:max_x, None]

				link[dummy_image == 1, :] = 1

				edge_contour, heirarchy = cv2.findContours(dummy_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
				if len(edge_contour) == 0:
					continue

				edge_contour = edge_contour[0].reshape([edge_contour[0].shape[0], 2])

				link = self.profiler(self._get_link, edge_contour, link, dummy_image)

		# if neighbouring pixel is positive, since loop only in positive pixels, link between them is made positive

		if self.config['weight_bbox'] and ret_all:

			if len(contours) != 0:
				weight = weight * np.where(weight != 0)[0].shape[0] / len(contours)

		if self.config['weight_bbox'] and self.config['link']:

			if ret_all:
				return (link * 255).astype(np.uint8), (target * 255).astype(np.uint8), weight
			else:
				return (target * 255).astype(np.uint8)

		if self.config['weight_bbox']:

			if ret_all:
				return (target * 255).astype(np.uint8), weight
			else:
				return (target * 255).astype(np.uint8)

		if self.config['link']:

			if ret_all:
				return (link * 255).astype(np.uint8), (target * 255).astype(np.uint8)
			else:
				return (target * 255).astype(np.uint8)

		return (target * 255).astype(np.uint8)

	def gradient_image(self, img):

		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

		return np.concatenate((gx, gy), axis=2)

	def remove_blank_annot(self, weight, remove_contour):

		if len(remove_contour) != 0:
			cv2.drawContours(weight, remove_contour, -1, 0, cv2.FILLED)

		return weight

	# Returns weights after making regions with no contours zero weight

	def getitem(self, index, type_test=None):

		return self.profiler(self.custom_get, index, type_test=type_test)

	def __len__(self):

		if self.Type == 'train':
			return len(self.images) // self.batchsize
		else:
			return min(len(self.images) // self.batchsize, 1000 // self.batchsize)

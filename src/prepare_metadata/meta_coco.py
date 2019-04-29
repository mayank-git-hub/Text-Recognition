import json
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pickle
from shutil import copyfile
import re


class MetaCoco():

	def __init__(self, config):

		self.config = config
		self.seed()

	def seed(self):

		np.random.seed(self.config['seed'])

	def create_annot(self, check_path=False, close_on_not_found=False, store_w_h=False, mask_bbox='mask'):

		#JSON: Java script object notation, it is a way to represent objects in the form of text!!!!

		num_unique = 0
		total_text = 0

		with open(self.config['metadata']['COCO']['meta']+'/cocotext.v2.json', 'r') as f: #READ MODE
			data = json.load(f)

			#Convert JSON for python use

		# json structure: {'imgs':{'img_id': {'height':, 'file_name': 'name', 'width':, 'set': 'val', 'id': 347319}},
		# 'anns':{'annot_id': {'area': 278.66, 'bbox': [474.9, 421.0, 17.7, 16.6], 'image_id': 71818, 'utf8_string': '15', 'class': 'machine printed', 'legibility': 'legible', 'mask': [474.9, 421.6, 475.3, 437.6, 492.6, 437.1, 492.3, 421.0], 'language': 'english', 'id': annot_id}},
		# 'imgToAnns:{'486372': [154148, 154149, 154150]'}, ''}
			
		annots = {}

		print('Length of data["anns"] = ', len(data['anns']), 'Length of data["imgs"] = ', len(data['imgs']))

		if self.config['metadata']['COCO']['background_all']:

			for key, value in data['imgs'].items():

				num_unique += 1
				path = ''
				for i in range(12 - len(str(value['id']))):
					path += '0'
				path += str(value['id'])
				path = 'COCO_train2014_'+path

				if check_path:

					if os.path.exists(self.config['metadata']['COCO']['dir']+'/'+path+'.jpg'):

						im=Image.open(self.config['metadata']['COCO']['dir']+'/'+path+'.jpg')
						path += '.jpg'

					elif os.path.exists(self.config['metadata']['COCO']['dir']+'/'+path+'.png'):

						im=Image.open(self.config['metadata']['COCO']['dir']+'/'+path+'.png')
						path += '.png'

					else:
						print(path)
						print('Image not found')

						if close_on_not_found:
							exit(0)

					w= int(im.size[0])
					h= int(im.size[1])

				
				if mask_bbox == 'mask':

					if store_w_h:
						annots[int(value['id'])] = {'path' : path, 'mask': [], 'text': [], 'size_w_h':[w, h]}
					else:
						annots[int(value['id'])] = {'path' : path, 'mask': [], 'text': []}

				else:

					if store_w_h:
						annots[int(value['id'])] = {'path' : path, 'bbox': [], 'text': [], 'size_w_h':[w, h]}
					else:
						annots[int(value['id'])] = {'path' : path, 'bbox': [], 'text': []}

			for key, value in data['anns'].items():

				# Example of key:value pair
				# {'annot_id': {'area': 278.66, 'bbox': [474.9, 421.0, 17.7, 16.6], 'image_id': 71818, 'utf8_string': '15', 'class': 'machine printed', 'legibility': 'legible', 'mask': [474.9, 421.6, 475.3, 437.6, 492.6, 437.1, 492.3, 421.0], 'language': 'english', 'id': annot_id}}

				total_text += 1

				if mask_bbox == 'mask':

					cur_bbox = list(np.array(value['mask']).astype(np.int32))				
					annots[int(value['image_id'])]['mask'].append(cur_bbox)

				else:

					cur_bbox = list(np.array(value['bbox']).astype(np.int32))				
					annots[int(value['image_id'])]['bbox'].append(cur_bbox)

				annots[int(value['image_id'])]['text'].append(value['utf8_string'])
		
		else:

			for key, value in data['anns'].items():

				# Example of key:value pair
				# {'annot_id': {'area': 278.66, 'bbox': [474.9, 421.0, 17.7, 16.6], 'image_id': 71818, 'utf8_string': '15', 'class': 'machine printed', 'legibility': 'legible', 'mask': [474.9, 421.6, 475.3, 437.6, 492.6, 437.1, 492.3, 421.0], 'language': 'english', 'id': annot_id}}

				if int(value['image_id']) in annots:

					total_text += 1

					if mask_bbox == 'mask':

						cur_bbox = list(np.array(value['mask']).astype(np.int32))				
						annots[int(value['image_id'])]['mask'].append(cur_bbox)

					else:

						cur_bbox = list(np.array(value['bbox']).astype(np.int32))				
						annots[int(value['image_id'])]['bbox'].append(cur_bbox)

					annots[int(value['image_id'])]['text'].append(value['utf8_string'])

				else:
					num_unique += 1
					total_text += 1
					path = ''
					for i in range(12 - len(str(value['image_id']))):
						path += '0'
					path += str(value['image_id'])
					path = 'COCO_train2014_'+path

					if check_path:

						if os.path.exists(self.config['dir']+'/'+path+'.jpg'):

							im=Image.open(self.config['dir']+'/'+path+'.jpg')
							path += '.jpg'

						elif os.path.exists(self.config['dir']+'/'+path+'.png'):

							im=Image.open(self.config['dir']+'/'+path+'.png')
							path += '.png'

						else:
							print(path)
							print('Image not found')

							if close_on_not_found:
								exit(0)

						w= int(im.size[0])
						h= int(im.size[1])

					cur_text = value['utf8_string']
					
					if mask_bbox == 'mask':

						cur_bbox = list(np.array(value['mask']).astype(np.int32))

						if store_w_h:
							annots[int(value['image_id'])] = {'path' : path, 'mask': [cur_bbox], 'text': [cur_text], 'size_w_h':[w, h]}
						else:
							annots[int(value['image_id'])] = {'path' : path, 'mask': [cur_bbox], 'text': [cur_text]}

					else:

						cur_bbox = list(np.array(value['bbox']).astype(np.int32))

						if store_w_h:
							annots[int(value['image_id'])] = {'path' : path, 'bbox': [cur_bbox], 'text': [cur_text], 'size_w_h':[w, h]}
						else:
							annots[int(value['image_id'])] = {'path' : path, 'bbox': [cur_bbox], 'text': [cur_text]}

		for i in annots.keys():

			if os.path.exists(self.config['metadata']['COCO']['label']+'/'+annots[i]['path']+'.pkl'):
				os.remove(self.config['metadata']['COCO']['label']+'/'+annots[i]['path']+'.pkl')

			f = open(self.config['metadata']['COCO']['label']+'/'+annots[i]['path']+'.pkl', 'wb')

			# /home/mayank/Desktop/GitRepos/Text/Segmentation/Dataset/labels_text/train2014

			all_annots = []

			if mask_bbox == 'mask':

				for j in annots[i]['mask']:

					j = np.array([[j[2*k],j[2*k+1]] for k in range(len(j)//2)]).astype(np.int32)
					all_annots.append(j.reshape(j.shape[0], 1, 2))

			else:

				for j in annots[i]['bbox']:
					j = np.array([[j[0], j[1]], [j[0]+j[2], j[1]], [j[0]+j[2], j[1]+j[3]], [j[0], j[1]+j[3]]])
					all_annots.append(j.reshape([4, 1, 2]))

			pickle.dump([all_annots, annots[i]['text']], f)

		print('Unique images = ', num_unique)
		print('Total text instances = ', total_text)

		self.split(annots)

	def split(self, annots):

		split_r = float(self.config['metadata']['COCO']['split_type'])

		# Creating a list of filenames for which annotation is available

		list_all = []

		for i in annots.keys():

			list_all.append(annots[i]['path'])

		idx = np.arange(len(list_all))
		np.random.shuffle(idx)

		# /home/mayank/Desktop/GitRepos/Text/Segmentation/Dataset/labels_text/meta

		if os.path.exists(self.config['metadata']['COCO']['meta']+'/train_files_'+str(split_r)+'.txt'):
			os.remove(self.config['metadata']['COCO']['meta']+'/train_files_'+str(split_r)+'.txt')

		f = open(self.config['metadata']['COCO']['meta']+'/train_files_'+str(split_r)+'.txt', 'w')

		train = np.array(list_all)[idx[:int(split_r*len(list_all))]]
		val = np.array(list_all)[idx[int(split_r*len(list_all)):]]
		for i in train:
			f.write(i+'\n')

		if os.path.exists(self.config['metadata']['COCO']['meta']+'/test_files_'+str(split_r)+'.txt'):
			os.remove(self.config['metadata']['COCO']['meta']+'/test_files_'+str(split_r)+'.txt')

		f = open(self.config['metadata']['COCO']['meta']+'/test_files_'+str(split_r)+'.txt', 'w')

		for i in val:
			f.write(i+'\n')

		if self.config['metadata']['COCO']['cal_avg']:

			self.calc_average(train)

	def calc_average(self, train):

		image_average = [np.zeros(3), 0]
		image_std = [np.zeros(3), 0]

		random_images = np.random.choice(train, min(1000, len(train)), replace=False)

		for i in random_images:

			image = plt.imread(self.config['metadata']['COCO']['image']+'/'+i)
			if len(image.shape) == 2:
				continue
			image_average[0] += np.sum(image, axis=(0, 1))
			image_average[1] += image.shape[0]*image.shape[1]

		image_average[0] = image_average[0]/image_average[1]

		for i in random_images:

			image = plt.imread(self.config['metadata']['COCO']['image']+'/'+i)
			if len(image.shape) == 2:
				continue
			image_std[0] += np.sum(np.square(image - image_average[0]), axis=(0, 1))
			image_std[1] += image.shape[0]*image.shape[1]

		image_std[0] = image_std[0]/image_std[1]

		with open(self.config['metadata']['COCO']['meta']+'/normalisation_'+str(train_ratio)+'.pkl', 'wb') as f:
			pickle.dump({'average': image_average, 'std': image_std}, f)

		#This function reads the images from JSON file format and splits into testing and training images

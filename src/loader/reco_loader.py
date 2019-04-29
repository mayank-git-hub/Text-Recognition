from .generic_dataloader import own_DataLoader
import numpy as np
import random
import torch
import cv2
import matplotlib.pyplot as plt
import imutils
from PIL import Image
import os
import pickle
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import transform as tf
from math import floor, ceil
import random
from .art import ArtificialGen

class RecoDataloader(own_DataLoader):

	def __init__(self, config, type_, channels, profiler = None, **kwargs):

		super().__init__(config, type_, profiler = profiler, **kwargs)
		self.seed()

		self.vertical_flip_prob = self.config['augmentation']['vertical_flip_prob']
		self.horizontal_flip_prob = self.config['augmentation']['horizontal_flip_prob']

		self.distort_prob = config['augmentation']['distort_prob']
		self.hori_tile, self.verti_tile = config['augmentation']['horizontal_tiles'], config['augmentation']['vertical_tiles'] 
		self.distort_mag = config['augmentation']['magnitude']

		self.shear_prob = config['augmentation']['shear_prob']
		self.afine_tf = tf.AffineTransform(shear = config['augmentation']['shear_mag'])

		if type_ == 'train':
			self.art_generator = ArtificialGen(config, self.all_characters)
			self.art_prob = config['augmentation']['art_prob_train']
		elif type_ == 'test':
			self.art_generator = ArtificialGen(config, self.all_characters)
			self.art_prob = config['augmentation']['art_prob_train']
			self.art_prob_test = config['augmentation']['art_prob_test']
			# self.char_distri = dict(zip(self.all_characters, [0]*len(self.all_characters)))

		self.num_channels = channels
		print("Working on",channels,"channeled images for",self.Type)

		if config['varying_width']:
			self.list_or_tensor = 'list'
		else:
			self.list_or_tensor = 'tensor'

		if self.Type != 'test_one' and 'OWN' in self.datasets and len(self.datasets) == 1 and self.Type == 'train':
			#do something
			self.only_OWN = True
			if os.path.exists(self.config['cache_path']+'/OWN_'+self.Type+'_cache.pkl'):
				with open(self.config['cache_path']+'/OWN_'+self.Type+'_cache.pkl', 'rb') as f:
					self.own_cache = pickle.load(f)
					# print(len(self.own_cache['images']))
			else:
				self.make_own_cache()			
		else:
			self.only_OWN = False

		if self.Type == 'test':
			if self.config['grayscale']:
				c = '_1'
			else:
				c = '_3'
			path = self.config['cache_path'] + '/' + '_'.join(self.datasets)+'_'+self.Type+c+'_'+self.config['project']+'.pkl'
			if os.path.exists(path):
				with open(path, 'rb') as f:
					self.test_cache = pickle.load(f)
					print("Loaded test cache")
				# print("Character Distribution:\n",self.test_cache['char_distribution'],'\n')
			else:
				self.make_test_cache()

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def get_abc(self):
		return self.abc

	def generate_string(self):
		return ''.join(random.choice(self.abc) for _ in range(self.seq_len))

	def resize(self, to_resize_batch, targets=None, fixed='fixed', height=32):

		all_images = []
		all_targets = []

		for i, sample in enumerate(to_resize_batch):
			# if sample.shape[0] < self.config['min_h'] and targets!=None:
			# 	# print("Skipping")
			# 	continue
			if fixed == 'fixed':
				size = (320, 32)
			else:
				new_width = int(max(int(height/sample.shape[0]*sample.shape[1]), 2*height)*1.5)
				size=(new_width, height)
				# print("New size:",size)
				
			if self.num_channels == 1:
				image = Image.fromarray(sample).convert('L')
			else:
				image = Image.fromarray(sample).convert('RGB')

			if self.config['augmentation']['flag'] and self.Type == 'train' and not self.only_OWN:
				#Random rotate to account for flipped words
				# if np.random.random() < self.vertical_flip_prob:
				# 	image = image.rotate(180)
				# if np.random.random() < self.horizontal_flip_prob:
				# 	image = image.transpose(Image.FLIP_LEFT_RIGHT)
				#Distortion

				if np.random.random() < self.distort_prob:
					hori_tile_final = self.hori_tile + np.random.randint(5) - 2
					verti_tile_final = self.verti_tile + np.random.randint(3) - 1
					mag_final = self.distort_mag + np.random.randint(3) - 1
					image = self.elastic_deformation(image, hori_tile_final, verti_tile_final, mag_final)
					# print('elastic')
				#Shearing
				# if np.random.random() < self.shear_prob:
				# 	image = tf.warp(np.array(image), inverse_map=self.afine_tf)
				# 	print(image.size)
				# 	print('shear')

			# print('There')
			if self.num_channels == 1:
				image = torch.FloatTensor(np.array(image.resize(size)).astype(np.uint8)[:,:,None].transpose(2, 0, 1)[None])/255
			else:
				image = torch.FloatTensor(np.array(image.resize(size)).astype(np.uint8).transpose(2, 0, 1)[None])/255

			all_images.append(image)
			all_targets.append(targets[i])

		return all_images, all_targets
		# else:
		# 	return all_images

	def rotate(self, cnt, image):

		#Consider some outer region for accurate text
		boundary = self.config['around_bound']
		#Find rotated rect's centre, rotation, heigh, width
		rect = list(cv2.minAreaRect(cnt))
		rect[1] = list(rect[1])
		#Add brder pixels
		rect[1][0] += boundary
		rect[1][1] += boundary
		rect[1] = tuple(rect[1])
		cnt = cv2.boxPoints(tuple(rect)).astype(np.int32).reshape([4, 1, 2])
		#Calculate minimum and maximum of contour coordinates for cropping
		min_x, min_y = np.min(cnt[:, 0, :], axis=0)
		
		min_x = max(0, min_x)
		min_y = max(0, min_y)

		max_x, max_y = np.max(cnt[:, 0, :], axis=0)

		max_x = min(image.shape[1], max_x)
		max_y = min(image.shape[0], max_y)


		if max_y - min_y <= 0 or max_x - min_x <= 0:
			return None

		mask = np.zeros([max_y - min_y, max_x - min_x, 3]).astype(np.uint8)
		#Shift origin
		cnt = cnt-np.array([min_x, min_y])
		cnt = cnt.astype(np.int32)
		#Draw just the rectangle on the mask
		cv2.drawContours(mask, [cnt], -1, (1, 1, 1), cv2.FILLED)
		#Crop part of the image with the contour only
		cropped_image = image[min_y:max_y, min_x:max_x, :].copy()
		cropped_image *= mask

		if rect[1][0]>rect[1][1]:
			rotated = imutils.rotate(cropped_image, rect[2])
			width, height = np.array(rect[1][0]).astype(np.int32), np.array(rect[1][1]).astype(np.int32)
		else:
			rotated = imutils.rotate(cropped_image, 90 + rect[2])
			width, height = np.array(rect[1][1]).astype(np.int32), np.array(rect[1][0]).astype(np.int32)

		center_x, center_y = np.array(rect[0]).astype(np.int32)-np.array([min_x, min_y])

		if self.num_channels == 1:
			return cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

		return rotated


	def custom_get(self, index):

		images = []
		target = []

		# print(len(self.own_cache['images']), len(self.own_cache['labels']))

		if self.only_OWN:
			
			if self.Type == 'train':
				random_images = self.profiler(np.random.choice, len(self.own_cache['images']), self.batchsize)
			else:
				# Not so random images(For testing)
				random_images = np.arange(index*self.batchsize,  min((index + 1)*self.batchsize, self.__len__()*self.batchsize))			

			for ind in random_images:
				images.append(self.own_cache['images'][ind])
				target.append(self.own_cache['labels'][ind])

			# print(len(images), len(target))
			# exit(0)
			if self.list_or_tensor == 'tensor':

				images = torch.cat(images, dim=0)
				seq_len = torch.IntTensor([target_i.shape[0] for target_i in target])
				seq = torch.IntTensor([])
				for i in range(len(target)):

					seq  = torch.cat([seq, target[i]], dim=0)

				sample = {"img": images, "seq": seq, "seq_len": seq_len, "aug": True}

			else:
			
				seq_len = [torch.IntTensor([target_i.shape[0]]) for target_i in target]
			
				seq = []
				for i in range(len(target)):

					seq.append(target[i])

				sample = {"img": images, "seq": seq, "seq_len": seq_len, "aug": True}

			# print(images.shape)
			# exit(0)

			return sample

		if self.Type == 'test':

			number = self.batchsize*self.art_prob_test
			images, target = self.art_gen(number)
			images, target = self.resize(images, target)
			cur_images = np.arange(index*(self.batchsize-number),  min((index + 1)*(self.batchsize-number), self.__len__())).astype(np.uint8)			
			# print(cur_images)
			for ind in cur_images:
				images.append(self.test_cache['images'][ind])
				target.append(self.test_cache['labels'][ind])

			# print(len(images), len(target))
			# exit(0)
			if self.list_or_tensor == 'tensor':

				images = torch.cat(images, dim=0)
				seq_len = torch.IntTensor([target_i.shape[0] for target_i in target])
				seq = torch.IntTensor([])
				for i in range(len(target)):

					seq = torch.cat([seq, target[i]], dim=0)

				sample = {"img": images, "seq": seq, "seq_len": seq_len, "aug": True}

			else:
			
				seq_len = [torch.IntTensor([target_i.shape[0]]) for target_i in target]
			
				seq = []
				for i in range(len(target)):
					seq.append(target[i])


				sample = {"img": images, "seq": seq, "seq_len": seq_len, "aug": True}
			return sample

		else:
			#TRAIN
			number = self.batchsize*self.art_prob
			images, target = self.profiler(self.art_gen, number)

			while True:
				if len(images) == self.batchsize:
					break
				#Get images from art
				# number = np.random.randint(self.batchsize*self.art_prob)
				random_images = self.profiler(np.random.choice, len(self.images), self.batchsize-len(images))

				for random_i in random_images:			

					for dataset_name, d in self.datasets_attr.items():
						# print(d)
						if random_i < d['range'][1] and random_i >= d['range'][0]:
							image_root = d['image_root']
							d_name = dataset_name
							# print(image_root)
							# print(dataset_name, i)
							break
					try:
						image = np.array(self.profiler(self.loader, image_root+'/'+self.images[random_i]))
						not_blank = [i for i in range(len(self.texts[random_i])) if len(self.texts[random_i][i])!=0]
					except:
						print('Errorenous Images', random_i)
						continue

					for (cnt, text) in zip(self.annots[random_i][not_blank], np.array(self.texts[random_i])[not_blank]):

						rotated = self.profiler(self.rotate, cnt, image)

						if rotated is None:
							# print("Rotated is none")
							continue
						images.append(rotated)
						target.append(torch.IntTensor(text))

						if len(images) == self.batchsize:
							break

					if len(images) == self.batchsize:
						break

			# print('Len of images', len(images), 'Before')

			if self.list_or_tensor == 'list':
				images, targets = self.profiler(self.resize, images, target, 'not_fixed')
			else:
				images, targets = self.profiler(self.resize, images, target, 'fixed')

			# print('Len of images', len(images), 'After')
			

			if self.list_or_tensor == 'tensor':

				images = torch.cat(images, dim=0)
				seq_len = torch.IntTensor([target_i.shape[0] for target_i in targets])
				seq = torch.IntTensor([])

				for i in range(len(targets)):

					seq  = torch.cat([seq, targets[i]], dim=0)

				sample = {"img": images, "seq": seq, "seq_len": seq_len, "aug": True}

			else:
			
				seq_len = [torch.IntTensor([target_i.shape[0]]) for target_i in targets]
			
				seq = []
				for i in range(len(targets)):
					seq.append(targets[i])
				# print(seq, seq_len)

				# print('Len of images: ', len(images))
				sample = {"img": images, "seq": seq, "seq_len": seq_len, "aug": True}

			return sample

	def getitem(self, index):

		return self.profiler(self.custom_get, index)


	def elastic_deformation(self, image, hori, verti, mag):
		"""
		Performs a random, elastic distortion on an image.

		This function performs a randomised, elastic distortion controlled
		by the parameters specified. The grid width and height controls how
		fine the distortions are. Smaller sizes will result in larger, more
		pronounced, and less granular distortions. Larger numbers will result
		in finer, more granular distortions. The magnitude of the distortions
		can be controlled using magnitude. This can be random or fixed.

		Good values for parameters are between 2 and 10 for the grid
		width and height, with a magnitude of between 1 and 10. Using values
		outside of these approximate ranges may result in unpredictable
		behaviour.

		Distorts the passed image(s) according to the parameters supplied during
		instantiation, returning the newly distorted image.
		:param images: The image(s) to be distorted.
		:type images: List containing PIL.Image object(s).
		:return: The transformed image(s) as a list of object(s) of type
		 PIL.Image.
		"""
		w, h = image.size

		horizontal_tiles = hori
		vertical_tiles = verti
		magnitude = mag

		width_of_square = int(floor(w / float(horizontal_tiles)))
		height_of_square = int(floor(h / float(vertical_tiles)))

		width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
		height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

		dimensions = []

		for vertical_tile in range(vertical_tiles):
			for horizontal_tile in range(horizontal_tiles):
				if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
					dimensions.append([horizontal_tile * width_of_square,
									   vertical_tile * height_of_square,
									   width_of_last_square + (horizontal_tile * width_of_square),
									   height_of_last_square + (height_of_square * vertical_tile)])
				elif vertical_tile == (vertical_tiles - 1):
					dimensions.append([horizontal_tile * width_of_square,
									   vertical_tile * height_of_square,
									   width_of_square + (horizontal_tile * width_of_square),
									   height_of_last_square + (height_of_square * vertical_tile)])
				elif horizontal_tile == (horizontal_tiles - 1):
					dimensions.append([horizontal_tile * width_of_square,
									   vertical_tile * height_of_square,
									   width_of_last_square + (horizontal_tile * width_of_square),
									   height_of_square + (height_of_square * vertical_tile)])
				else:
					dimensions.append([horizontal_tile * width_of_square,
									   vertical_tile * height_of_square,
									   width_of_square + (horizontal_tile * width_of_square),
									   height_of_square + (height_of_square * vertical_tile)])

		# For loop that generates polygons could be rewritten, but maybe harder to read?
		# polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

		# last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
		last_column = []
		for i in range(vertical_tiles):
			last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

		last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

		polygons = []
		for x1, y1, x2, y2 in dimensions:
			polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

		polygon_indices = []
		for i in range((vertical_tiles * horizontal_tiles) - 1):
			if i not in last_row and i not in last_column:
				polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

		def do(image):

			for a, b, c, d in polygon_indices:
				dx = random.randint(-magnitude, magnitude)
				dy = random.randint(-magnitude, magnitude)

				x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
				polygons[a] = [x1, y1,
							   x2, y2,
							   x3 + dx, y3 + dy,
							   x4, y4]

				x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
				polygons[b] = [x1, y1,
							   x2 + dx, y2 + dy,
							   x3, y3,
							   x4, y4]

				x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
				polygons[c] = [x1, y1,
							   x2, y2,
							   x3, y3,
							   x4 + dx, y4 + dy]

				x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
				polygons[d] = [x1 + dx, y1 + dy,
							   x2, y2,
							   x3, y3,
							   x4, y4]

			generated_mesh = []
			for i in range(len(dimensions)):
				generated_mesh.append([dimensions[i], polygons[i]])


			try:
				return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)
			except:
				print('Image size really small')
				plt.imshow(image)
				plt.show()
				exit(0)

		if len(image.size) == 2:
			
			return do(image)

		else:

			if image.shape[2] == 1:
				z = np.zeros(image.size[1], image.size[0], 1)
				z[:,:,0] = do(image)
				return z

			else:
				#3-channeled
				z = np.zeros(image.size[1], image.size[0], 3)
				for i in range(3):
					z[:,:,i] = do(image[:,:,i])

				return z

	def __len__(self):

		if self.only_OWN:
			return len(self.own_cache['images']) // self.batchsize

		else:
			if self.Type == 'test':
				return len(self.test_cache['images'])
			else:
				return len(self.images)//self.batchsize

	def art_gen(self, number):
		#Append number of art images to images and target to target
		added = 0
		images, target = [], []

		while added < number:
			img, text = self.art_generator.get_image()
			if img != None:
				added += 1
				target.append(torch.IntTensor([self.char_to_encoding[c] for c in text]))
				images.append(np.array(img))

		return images, target

	def make_own_cache(self):

		d_name, d, image_root = 'OWN', self.datasets_attr['OWN'], self.datasets_attr['OWN']['image_root']
		self.own_cache = {'l_to_img_name':{}}
		images, target = [], []

		for i in range(len(self.images)):

			if not (i < d['range'][1] and i >= d['range'][0]):
				continue	

			try:
				image = np.array(self.loader(image_root+'/'+self.images[i]))

				not_blank = [j for j in range(len(self.texts[i])) if len(self.texts[i][j])!=0]
			
			except:
				continue

			for (cnt, text) in zip(self.annots[i][not_blank], np.array(self.texts[i])[not_blank]):

				# print(cv2.contourArea(cnt), len(text))
				# if len(text)<3:#cv2.contourArea(cnt) < 400*len(text) or
				# 	continue

				rotated = self.rotate(cnt, image)
				if rotated is None:
					continue

				images.append(rotated)
				l = [self.encoding_to_char[e] for e in text]
				self.own_cache['l_to_img_name'][''.join(l)] = self.images[i]
				target.append(torch.IntTensor(text))

		if self.list_or_tensor == 'list':
			images, targets = self.resize(images, target, 'not_fixed')
		else:
			images, targets = self.resize(images, target, 'fixed')

		self.own_cache['images'] = images
		self.own_cache['labels'] = targets
		# print(len(targets))
		dump_name = self.config['cache_path']+'/OWN_'+self.Type+'_cache.pkl'

		with open(dump_name, 'wb') as f:
			pickle.dump(self.own_cache, f)

		print("Successfully dumped",self.Type,"cache for OWN exclusive")


	def make_test_cache(self):

		self.test_cache = {'l_to_img_name':{}}
		images, target = [], []
		# self.char_distri = dict(zip(self.all_characters, [0]*len(self.all_characters)))
		print("***Making test cache***")
		for i in range(len(self.images)):

			for dataset_name, d in self.datasets_attr.items():
					# print(d)
				if i < d['range'][1] and i >= d['range'][0]:
					image_root = d['image_root']
					d_name = dataset_name
					# print(image_root)
					# print(dataset_name, i)
					break
			try:
				image = np.array(self.loader(image_root+'/'+self.images[i]))
				not_blank = [j for j in range(len(self.texts[i])) if len(self.texts[i][j])!=0]			
			except:
				continue

			for (cnt, text) in zip(self.annots[i][not_blank], np.array(self.texts[i])[not_blank]):

				# print(cv2.contourArea(cnt), len(text))
				if len(text)<3:#cv2.contourArea(cnt) < 400*len(text) or 
					continue

				rotated = self.rotate(cnt, image)
				if rotated is None:
					continue

				# s = [self.encoding_to_char[c] for c in text]
				# for charac in s:
				# 	self.char_distri[charac] += 1

				images.append(rotated)
				l = [self.encoding_to_char[e] for e in text]
				self.test_cache['l_to_img_name'][''.join(l)] = self.images[i]
				target.append(torch.IntTensor(text))

		if self.list_or_tensor == 'list':
			images, targets = self.resize(images, target, 'not_fixed')
		else:
			images, targets = self.resize(images, target, 'fixed')

		self.test_cache['images'] = images
		self.test_cache['labels'] = targets
		# self.test_cache['char_distribution'] = self.char_distri

		# print("Distribution of characters:\n",self.char_distri,'\n')

		if self.config['grayscale']:
			c = '_1'
		else:
			c = '_3'
		dump_name = self.config['cache_path'] + '/' + '_'.join(self.datasets)+'_'+self.Type+c+'_'+self.config['project']+'.pkl'

		with open(dump_name, 'wb') as f:
			pickle.dump(self.test_cache, f)

		print("Successfully dumped test cache for",'_'.join(self.datasets))
		print('Total # testing examples is',len(self.test_cache['images']))
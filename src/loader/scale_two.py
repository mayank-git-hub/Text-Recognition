from .generic_dataloader import own_DataLoader
import numpy as np
import torch
from PIL import Image
import cv2
import threading
import matplotlib.pyplot as plt
import random
class scale_two(own_DataLoader):

	def __init__(self, config, Type, profiler = None, **kwargs):

		super().__init__(config, Type, profiler = profiler, **kwargs)
		self.seed()

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def weight_thresh(self, weight):

		weight[weight<=self.config['weight_threshold']] = self.config['weight_threshold']

		return weight
	
	def aspect_resize(self, image, contour, remove_contour):

		image, image_shape, row, column, ratio = self._32_2_aspect_resize(image)
		contour = self.get_resized_contours(row, column, image_shape, contour.copy(), ratio)
		remove_contour = self.get_resized_contours(row, column, image_shape, remove_contour.copy(), ratio)
		link, target, weight = self.get_link(contour, row, column, image_shape, [image.size[1], image.size[0]])
		weight = self.remove_blank_annot(weight, remove_contour)
		weight = self.weight_thresh(weight)

		return image, link, Image.fromarray(target).convert('L'), weight, contour

		#Returns resized image after calling aspect resize to nearest 32
		#contour and contours to be removed resized within this function after calls to respective functions
		#weight matrix returned

	def _32_2_aspect_resize(self, image):

		image = np.array(image)

		max_edge = max(image.shape[0]//2, image.shape[1]//2)
		ratio = 1
		if max_edge >=self.config['max_size']:
			ratio = max_edge/self.config['max_size']

		to_resize_r, to_resize_h = int(image.shape[0]/ratio/2), int(image.shape[1]/ratio/2)		
		image = np.array(Image.fromarray(image).convert('RGB').resize(size=(to_resize_h, to_resize_r)))
		
		image_shape = image.shape

		row, column = (to_resize_r//32)*32 + 32, (to_resize_h//32)*32 + 32

		blank = (np.ones([row, column, 3])*np.mean(image)).astype(np.uint8)
		blank[(row - image.shape[0])//2:(row - image.shape[0])//2 + image.shape[0], (column - image.shape[1])//2:(column - image.shape[1])//2 + image.shape[1], :] = image
		
		return Image.fromarray(blank).convert('RGB'), image_shape, row, column, ratio

	def get_resized_contours(self, row, column, image_shape, contour, ratio_reshaping):

		contour = (contour/ratio_reshaping/2).astype(np.int32)
		shift = np.array([column - image_shape[1], row - image_shape[0]])/2
		count = 0
		total_length = contour.shape[0]

		for i in range(total_length):

			if np.any(contour[i- count].reshape([contour[i- count].shape[0], 2])[:, 0] > image_shape[1]) or np.any(contour[i- count].reshape([contour[i- count].shape[0], 2])[:, 1] > image_shape[0]):
				contour = np.delete(contour, [i- count], axis=0)
				count += 1
			else:
				contour[i- count] = ((contour[i - count].reshape([contour[i- count].shape[0], 2])+shift)[None, :]).reshape([contour[i- count].shape[0], 1, 2]).astype(np.int32)
				if cv2.contourArea(contour[i- count]) == 0:
					contour = np.delete(contour, [i- count], axis=0)
					count += 1

		return contour

	def inverse_resized_contours(self, row, column, image_shape, contour, ratio_reshaping):

		shift = np.array([column - image_shape[1], row - image_shape[0]])/2
		total_length = contour.shape[0]

		for i in range(total_length):

			contour[i] = (((contour[i].reshape([contour[i].shape[0], 2])-shift)[None, :]).reshape([contour[i].shape[0], 1, 2])*2*ratio_reshaping).astype(np.int32)

		return contour


	def rotate(self, image, target, link, contour, angle):

		def contour_rotate(contours_, angle, h, w):
			print(h,w)
			contours = np.copy(contours_)

			if angle == 0:
				
				return contours

			elif angle == 90:

				contours[:,:,:,0], contours[:,:,:,1] = contours[:,:,:,1], w-contours[:,:,:,0]

			elif angle == 180:

				x, y = contours[:,:,:,0], contours[:,:,:,1]
				contours[:,:,:,0] = x
				contours[:,:,:,1] = h-y

			elif angle == 270:

				x, y = contours[:,:,:,0], contours[:,:,:,1]
				contours[:,:,:,0] = h-y
				contours[:,:,:,1] = x

			return contours

		print('original:',contour, contour.shape)

		if angle == 0:
			return image, target, link, contour

		elif angle == 90:
			image = image.rotate(90)
			target = target.rotate(90)
			link = link.transpose((1,0,2))
			contour_f = contour_rotate(contour, 90, image.size[1], image.size[0])

		elif angle == 180:
			image = image.rotate(180)
			target = target.rotate(180)
			link = np.flip(link, axis=1)
			contour_f = contour_rotate(contour, 180, image.size[1], image.size[0])

		elif angle == 270:
			image = image.rotate(270)
			target = target.rotate(270)
			link = np.flip(link, axis=0).transpose((1,0,2))
			contour_f = contour_rotate(contour, 270, image.size[1], image.size[0])

		print('final:',contour_f, contour_f.shape)

		return image, target, link, contour_f

	def _custom_get(self, i, no, all_returns):

		for dataset_name, d in self.datasets_attr.items():
			if d['range'][0] <= i and d['range'][1] > i:
				image_root = d['image_root']
				d_name = dataset_name
				break

		image_new, link_new, target_new, weight_new, contour_i = self.aspect_resize(self.loader(image_root+'/'+self.images[i]), self.annots[i].copy(), self.remove_annots[i].copy())#, big_target_new
		image_new, target_new, link_new, contour_i = self.rotate(image_new, target_new, link_new, contour_i, 90)
		show = True
		if show:
			plt.imsave('img.png',image_new)
			num = np.array(image_new)
			cv2.drawContours(num, contour_i, -1, (0,255,0), 3)
			plt.imsave('contours.png',num)
			plt.imsave('target.png',target_new)
		img = self.transform(image_new).unsqueeze(0)
		target = self.target_transform(target_new).unsqueeze(0)
		link = self.target_transform(link_new).unsqueeze(0)
		weight = torch.FloatTensor(weight_new).unsqueeze(0).unsqueeze(0)
		all_returns[no] = [img, target, link, weight, contour_i, d_name]


	def custom_get(self, index):

		# img = torch.FloatTensor(np.zeros([self.batchsize, 3, self.image_size, self.image_size]))
		# target = torch.FloatTensor(np.zeros([self.batchsize, 1, self.image_size, self.image_size]))
		# link = torch.FloatTensor(np.zeros([self.batchsize, 8, self.image_size, self.image_size]))
		# weight = torch.FloatTensor(np.zeros([self.batchsize, 1, self.image_size, self.image_size]))

		img = []
		target = []
		link = []
		weight = []
		path = []
		contour = []
		text = []
		d_name_list = []

		if self.Type == 'train':
			random_images = self.profiler(np.random.choice, len(self.images), self.batchsize)
		else:
			# Not so random images(For testing)
			random_images = np.arange(index*self.batchsize,  min((index + 1)*self.batchsize, self.__len__()*self.batchsize))

		all_threads = []

		all_returns = {}

		for i in range(len(random_images)):

			all_returns[i] = []

		for no, i in enumerate(random_images):

			t1 = threading.Thread(target=self._custom_get, args=(i, no, all_returns))
			t1.start()
			all_threads.append(t1)

		for no, t1 in enumerate(all_threads):

			t1.join()
			img.append(all_returns[no][0])
			target.append(all_returns[no][1])
			link.append(all_returns[no][2])
			weight.append(all_returns[no][3])
			contour.append(all_returns[no][4])
			path.append(self.images[random_images[no]])
			text.append(self.texts[i])
			d_name_list.append(all_returns[no][5])

		return img, link, target, weight, contour, path, text, d_name_list

		#Returns data minibatch with target, links, contours weight etc. all attributes
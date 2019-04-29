from .generic_dataloader import own_DataLoader
import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
import cv2
import threading
import matplotlib.pyplot as plt
import random
from ..helper.utils import homographic_rotation


class DeteLoader(own_DataLoader):

	def __init__(self, config, type_, profiler = None, **kwargs):

		super().__init__(config, type_, profiler = profiler, **kwargs)
		self.seed()
		self.type_dete = ['1', '0.75', '1.25'] # 1 means [512, 768] # 

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def weight_thresh(self, weight):

		weight[weight<=self.config['weight_threshold']] = self.config['weight_threshold']

		return weight
	
	def aspect_resize(self, image, contour, remove_contour, type_):

		base_r, base_c = 512 + 64*2, 768 + 64*2
		row, column = int(base_r*float(type_)), int(base_c*float(type_))

		image, orig_shape = self._row_column_aspect_resize(row, column, image)
		contour = self._row_column_get_resized_contours(row, column, contour.copy(), orig_shape)
		remove_contour = self._row_column_get_resized_contours(row, column, remove_contour.copy(), orig_shape)

		# image, image_shape, row, column, ratio = self._32_by2_aspect_resize(image)
		# contour = self.get_resized_contours(row, column, image_shape, contour.copy(), ratio)
		# remove_contour = self.get_resized_contours(row, column, image_shape, remove_contour.copy(), ratio)
		
		if self.config['weight_bbox'] and self.config['link']:

			link, target, weight = self.get_link(contour, [image.size[1], image.size[0]])
			weight = self.remove_blank_annot(weight, remove_contour)
			weight = self.weight_thresh(weight)
			# print('Original',link.shape)

			return image, link, Image.fromarray(target).convert('L'), weight, contour

			#Returns resized image after calling aspect resize to nearest 32
			#contour and contours to be removed resized within this function after calls to respective functions
			#weight matrix returned

		elif self.config['weight_bbox']:

			target, weight = self.get_link(contour, [image.size[1], image.size[0]])
			weight = self.remove_blank_annot(weight, remove_contour)
			weight = self.weight_thresh(weight)

			return image, Image.fromarray(target).convert('L'), weight, contour

		elif self.config['link']:

			link, target = self.get_link(contour, [image.size[1], image.size[0]])

			return image, link, Image.fromarray(target).convert('L'), contour

		else:

			target = self.get_link(contour, [image.size[1], image.size[0]])	

			return image, Image.fromarray(target).convert('L'), contour	

	def _row_column_aspect_resize(self, row, column, image):

		image = np.array(image)
		mean = np.mean(image)
		orig_shape = image.shape

		max_edge = max(image.shape[0]/row, image.shape[1]/column)
		to_resize_r, to_resize_c = int(image.shape[0]/max_edge), int(image.shape[1]/max_edge)
		image = np.array(Image.fromarray(image).convert('RGB').resize(size=(to_resize_c, to_resize_r)))

		blank = (np.ones([row, column, 3])*mean).astype(np.uint8)
		blank[(row - image.shape[0])//2:(row - image.shape[0])//2 + image.shape[0], (column - image.shape[1])//2:(column - image.shape[1])//2 + image.shape[1], :] = image
		
		return Image.fromarray(blank).convert('RGB'), orig_shape

	def _row_column_get_resized_contours(self, row, column, contour, orig_shape):

		max_edge = max(orig_shape[0]/row, orig_shape[1]/column)
		to_resize_r, to_resize_c = int(orig_shape[0]/max_edge), int(orig_shape[1]/max_edge)
		count = 0
		shift = np.array([column - to_resize_c, row - to_resize_r])/2

		for i in range(len(contour)):
			contour[i] = np.array(contour[i])
			contour[i][:, :, 0] = contour[i][:, :, 0]/max_edge
			contour[i][:, :, 1] = contour[i][:, :, 1]/max_edge

		total_length = len(contour)

		for i in range(total_length):

			temp = ((contour[i - count].reshape([contour[i- count].shape[0], 2])+shift)[None, :]).reshape([contour[i- count].shape[0], 1, 2]).astype(np.int32)

			if cv2.contourArea(temp) == 0:
				contour = np.delete(contour, [i- count], axis=0)
				count += 1
			else:
				temp = temp.reshape([contour[i- count].shape[0], 2])
				temp[temp[:, 0]>=column] = column
				temp[temp[:, 1]>=row] = row
				contour[i- count] = temp.reshape([contour[i- count].shape[0], 1, 2])

		return contour

	def _row_column_inverse_resized_contours(self, row, column, contour, orig_shape):

		max_edge = max(orig_shape[0]/row, orig_shape[1]/column)
		to_resize_r, to_resize_c = int(orig_shape[0]/max_edge), int(orig_shape[1]/max_edge)
		shift = np.array([column - to_resize_c, row - to_resize_r])/2

		for i in range(len(contour)):

			contour[i] = contour[i].reshape([contour[i].shape[0], 2])-shift
			contour[i][:, 0] *= max_edge
			contour[i][:, 1] *= max_edge
			contour[i] = contour[i].reshape(contour[i].shape[0], 1, 2).astype(np.int32)

		return contour

	def contour_rotate(self, contours_, angle, h, w):
		contours = np.copy(contours_)

		if angle == 0:
			
			return contours

		elif angle == 90:
			
			for i in range(contours.shape[0]):
				x_orig = contours[i,:,:,0].copy()
				contours[i][:,:,0]  = contours[i][:,:,1]
				contours[i][:,:,1] = w-x_orig

		elif angle == 180:

			for i in range(contours.shape[0]):
				# print(contours[i])
				contours[i][:,:,1] = h - contours[i][:,:,1]
				contours[i][:,:,0] = w - contours[i][:,:,0]
			
		elif angle == 270:

			for i in range(contours.shape[0]):
				x_orig = contours[i][:,:,0].copy()
				contours[i][:,:,0] = h-contours[i][:,:,1]
				contours[i][:,:,1] = x_orig		

		return contours

	def rotate(self, image, target, link, contour, angle):

		if angle == 0:
			if link is None:
				return image, target, contour
			return image, target, link, contour

		elif angle == 90:
			image = image.transpose(Image.ROTATE_90)
			target = target.transpose(Image.ROTATE_90)
			link = np.rot90(link)

			contour_f = self.contour_rotate(contour, 90, image.size[0], image.size[1])

		elif angle == 180:

			image = image.transpose(Image.ROTATE_180)
			target = target.transpose(Image.ROTATE_180)
			link = np.rot90(np.rot90(link))
			contour_f = self.contour_rotate(contour, 180, image.size[1], image.size[0])

		elif angle == 270:
			image = image.transpose(Image.ROTATE_270)
			target = target.transpose(Image.ROTATE_270)
			link = np.rot90(np.rot90(np.rot90(link)))
			contour_f = self.contour_rotate(contour, 270, image.size[0], image.size[1])

		if link is None:
				return image, target, contour_f

		return image, target, link, contour_f

	def _custom_get(self, i, no, type_, all_returns, x1=None, y2=None, z3=None, index=None):

		flag = self.config['aug_flag']
		angle = np.random.choice([0,90,180,270], p = self.config['augmentation']['rotation'])

		for dataset_name, d in self.datasets_attr.items():
			if d['range'][0] <= i and d['range'][1] > i:
				image_root = d['image_root']
				d_name = dataset_name
				break

		if np.random.choice([0, 1], p=[self.config['augmentation']['grey_scale'], 1 - self.config['augmentation']['grey_scale']]) == 0 and self.Type=='train':

			image_ = Image.open(image_root+'/'+self.images[i]).convert('L').convert('RGB')
		else:

			image_ = self.loader(image_root+'/'+self.images[i])

		if np.random.choice([0, 1], p=[self.config['augmentation']['blur']['prob'], 1 - self.config['augmentation']['blur']['prob']]) == 0 and self.Type=='train':

			image_ = image_.filter(ImageFilter.GaussianBlur(radius=max(image_.size)/self.config['augmentation']['blur']['divide']))

		image_ = np.array(image_, np.uint8)

		if np.random.choice([0, 1], p=[self.config['augmentation']['intensity']['prob'], 1 - self.config['augmentation']['intensity']['prob']]) == 0 and self.Type=='train':

			image_ = image_.astype(np.float32)
			image_ = image_/image_.mean()*128
			image_ = (image_*(np.random.randint(0, (self.config['augmentation']['intensity']['max']-self.config['augmentation']['intensity']['min'])*10000)+self.config['augmentation']['intensity']['min']*10000) / 10000)
			image_[image_>255] = 255
			image_ = image_.astype(np.uint8)

		if np.random.choice([0, 1], p=[self.config['augmentation']['invert'], 1 - self.config['augmentation']['invert']]) == 0 and self.Type=='train':

			image_ = 255 - image_

		if np.random.choice([0, 1], p=[self.config['augmentation']['random_channel_shift'], 1 - self.config['augmentation']['random_channel_shift']]) == 0 and self.Type=='train':

			image_ = self.random_channel_shift(image_)

		contours_1, contours_2 = self.annots[i].copy(), self.remove_annots[i].copy()

		if x1 is not None and y2 is not None and z3 is not None and self.Type=='train':	

			# print(contours_1, contours_2)	
			image_, contour_list = homographic_rotation([x1, y2, z3], image_, [contours_1, contours_2], self.images[i], i)
			contours_1, contours_2 = contour_list

		image_ = Image.fromarray(image_)

		if self.config['weight_bbox'] and self.config['link']:

			
			image_new, link_new, target_new, weight_new, contour_i = self.aspect_resize(image_, contours_1, contours_2, type_)
			if flag and self.Type == 'train':
				image_new, target_new, link_new, contour_i = self.rotate(image_new, target_new, link_new, contour_i, angle)
			img = self.transform(image_new).unsqueeze(0)
			target = self.target_transform(target_new).unsqueeze(0)
			link = self.target_transform(link_new.copy()).unsqueeze(0)
			weight = torch.FloatTensor(weight_new).unsqueeze(0).unsqueeze(0)
			if self.config['gradient']:
				all_returns[no] = [torch.cat([img, torch.FloatTensor(self.gradient_image(img[0].data.cpu().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)).unsqueeze(0)], dim=1), target, link, weight, contour_i, d_name]
			else:
				all_returns[no] = [img, target, link, weight, contour_i, d_name]

		elif self.config['weight_bbox']:

			image_new, target_new, weight_new, contour_i = self.aspect_resize(image_, contours_1, contours_2, type_)
			if flag and self.Type == 'train':
				image_new, target_new, contour_i = self.rotate(image_new, target_new, contour_i, angle)
			img = self.transform(image_new).unsqueeze(0)
			target = self.target_transform(target_new).unsqueeze(0)
			weight = torch.FloatTensor(weight_new).unsqueeze(0).unsqueeze(0)
			if self.config['gradient']:
				all_returns[no] = [torch.cat([img, torch.FloatTensor(self.gradient_image(img[0].data.cpu().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)).unsqueeze(0)], dim=1), target, weight, contour_i, d_name]
			else:
				all_returns[no] = [img, target, None, weight, contour_i, d_name]

		elif self.config['link']:

			image_new, link_new, target_new, contour_i = self.aspect_resize(image_, contours_1, contours_2, type_)
			if flag and self.Type == 'train':
				image_new, target_new, link_new, contour_i = self.rotate(image_new, target_new, link_new, contour_i, angle)
			img = self.transform(image_new).unsqueeze(0)
			target = self.target_transform(target_new).unsqueeze(0)
			link = self.target_transform(link_new.copy()).unsqueeze(0)
			if self.config['gradient']:
				all_returns[no] = [torch.cat([img, torch.FloatTensor(self.gradient_image(img[0].data.cpu().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)).unsqueeze(0)], dim=1), target, link, contour_i, d_name]
			else:
				all_returns[no] = [img, target, link, None, contour_i, d_name]

		else:

			image_new, target_new, contour_i = self.aspect_resize(image_, contours_1, contours_2, type_)
			if flag and self.Type == 'train':
				image_new, target_new, contour_i = self.rotate(image_new, target_new, contour_i, angle)
			img = self.transform(image_new).unsqueeze(0)
			target = self.target_transform(target_new).unsqueeze(0)

			if self.config['gradient']:
				all_returns[no] = [torch.cat([img, torch.FloatTensor(self.gradient_image(img[0].data.cpu().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)).unsqueeze(0)], dim=1), target, contour_i, d_name]
			else:
				all_returns[no] = [img, target, None, None, contour_i, d_name]

	def custom_get(self, index, type_test=None):

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

			random_images = self.profiler(np.random.choice, len(self.images), self.batchsize, p=self.prob_sample)

		else:
			# Not so random images(For testing)
			random_images = np.arange(index*self.batchsize,  min((index + 1)*self.batchsize, self.__len__()*self.batchsize))

		all_threads = []

		all_returns = {}

		show_now = False

		if show_now:

			for dataset_name, d in self.datasets_attr.items():
				if d['range'][0] <= random_images[0] and d['range'][1] > random_images[0]:
					image_root = d['image_root']
					d_name = dataset_name
					break

			img = plt.imread(image_root+'/'+self.images[random_images[0]])
			cv2.drawContours(img, self.annots[random_images[0]], -1, (0, 255, 0), 1)
			plt.imshow(img)
			plt.show()

		for i in range(len(random_images)):

			all_returns[i] = []

		for no, i in enumerate(random_images):
			if self.Type == 'train':
				x1 = np.random.randint(-30,30)
				y2 = np.random.randint(-30,30)
				z3 = np.random.randint(-30,30)
				t1 = threading.Thread(target=self._custom_get, args=(i, no, self.type_dete[np.random.randint(len(self.type_dete))], all_returns, x1, y2, z3, index))
			else:
				t1 = threading.Thread(target=self._custom_get, args=(i, no, type_test, all_returns))
			t1.start()
			all_threads.append(t1)

		for no, t1 in enumerate(all_threads):
			t1.join()

			img.append(all_returns[no][0])
			target.append(all_returns[no][1])
			link.append(all_returns[no][2])
			weight.append(all_returns[no][3])
			contour.append(all_returns[no][4])
			d_name_list.append(all_returns[no][5])
			path.append(self.images[random_images[no]])
			text.append(self.texts[i])
		
		return img, link, target, weight, contour, path, text, d_name_list


		#Returns data minibatch with target, links, contours weight etc. all attributes
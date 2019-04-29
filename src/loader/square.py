from .generic_dataloader import own_DataLoader
# from scipy.misc import imresize
import numpy as np
import torch
from PIL import Image
import cv2

class square(own_DataLoader):

	def __init__(self, config, Type, profiler = None, **kwargs):

		super().__init__(config, Type, profiler = profiler, **kwargs)

	def aspect_resize(self, image, contour, remove_contour):

		image, image_shape, row, column = self._aspect_resize(image)
		contour = self.get_resized_contours(row, column, image_shape, [image.size[1], image.size[0]], contour)
		remove_contour = self.get_resized_contours(row, column, image_shape, [image.size[1], image.size[0]], remove_contour)
		link, target, weight = self.get_link(contour, row, column, image_shape, [image.size[1], image.size[0]])
		weight = self.remove_blank_annot(weight, remove_contour)
		weight[weight<=self.config['weight_threshold']] = self.config['weight_threshold']

		return image, link, Image.fromarray(target).convert('L'), weight

	def _aspect_resize(self, image):

		image = np.array(image)
		r_h = max(image.shape[0], image.shape[1])
		blank = (np.ones([r_h, r_h, 3])*np.mean(image)).astype(np.uint8)
		blank[(r_h - image.shape[0])//2:(r_h - image.shape[0])//2 + image.shape[0], (r_h - image.shape[1])//2:(r_h - image.shape[1])//2 + image.shape[1], :] = image
		image_shape = image.shape
		# image = imresize(blank, (self.image_size, self.image_size))
		image = np.array(Image.fromarray(blank).resize(size=(self.image_size, self.image_size)))

		return Image.fromarray(image).convert('RGB'), image_shape, r_h, r_h

		#Forms a picture, such that image becomes a square with largest side as side, original image at centre surrounded padded by image.mean()

	def get_resized_contours(self, row, column, image_shape, resized_image_shape, contour):

		ratio = np.array([resized_image_shape[1]/column, resized_image_shape[0]/row])
		shift = np.array([column - image_shape[1], row - image_shape[0]])/2
		count = 0
		total_length = contour.shape[0]
		# show = False
		# if contour.shape[0] == 114:
		# 	print(ratio)
		# 	print(shift)
		# 	print(total_length)
		# 	show = True
		for i in range(total_length):
			# if show:
			# 	print(contour.shape)
			if np.any(contour[i- count].reshape([contour[i- count].shape[0], 2])[:, 0] > image_shape[1]) or np.any(contour[i- count].reshape([contour[i- count].shape[0], 2])[:, 1] > image_shape[0]):
				contour = np.delete(contour, [i- count], axis=0)
				count += 1
			else:
				contour[i- count] = ((contour[i - count].reshape([contour[i- count].shape[0], 2])+shift)*ratio[None, :]).reshape([contour[i- count].shape[0], 1, 2]).astype(np.int32)
				if cv2.contourArea(contour[i- count]) == 0:
					contour = np.delete(contour, [i- count], axis=0)
					count += 1

			# if show:
			# 	print(i, count)

		return contour

	def custom_get(self, index):

		img = []
		target = []
		link = []
		weight = []
		path = []
		contour = []
		text = []

		random_images = np.random.choice(len(self.images), self.batchsize)

		for no, i in enumerate(random_images):

			image_new, link_new, target_new, weight_new = self.aspect_resize(self.loader(self.image_root+'/'+self.images[i]), self.annots[i].copy(), self.remove_annots[i].copy())#, big_target_new
			img.append(self.transform(image_new).unsqueeze(0))
			target.append(self.target_transform(target_new).unsqueeze(0))
			link.append(self.target_transform(link_new).unsqueeze(0))
			weight.append(torch.FloatTensor(weight_new).unsqueeze(0).unsqueeze(0))

			contour.append(self.annots[i])
			path.append(self.images[i])
			text.append(self.texts[i])

		return img, link, target, weight, contour, path, text
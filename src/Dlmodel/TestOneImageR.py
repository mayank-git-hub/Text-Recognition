from .Dlmodel import Dlmodel
from ..helper.logger import Logger
from ..helper.utils import get_connected_components, strLabelConverter
import torch
import time
import random
import numpy as np
import editdistance
import os
import matplotlib.pyplot as plt

log = Logger()

class TestOneImageRClass(Dlmodel):

	def __init__(self, config, mode = 'test_one', profiler=None):
		
		super().__init__(config, mode, profiler)
		self.seed()
		self.converter = strLabelConverter(self.test_data_loader.get_abc())

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def test_one_image_with_cropped(self, image, path=None, save=True):

		self.profiler(self.start_testing)

		with torch.no_grad():

			resized_img = self.test_data_loader.resize([np.array(image)], fixed='not_fixed')[0][0]

			if self.cuda:
				resized_img = resized_img.cuda()

			out = self.profiler(self.model, resized_img)
			# print(out.shape)
			_, preds = out.max(2)
			preds = preds.transpose(1, 0).contiguous().view(-1)
			preds_size = torch.IntTensor([out.size(0)]).int()
			predicted_final_label = self.converter.decode(preds.data, preds_size.data, raw=False)
			# print('Predicted Label:',predicted_final_label)
			if predicted_final_label == '':
				predicted_final_label = 'NONE'
			# return out
			if save:
				path_to_save = path + '/' + predicted_final_label + '.png'
				plt.imsave(path_to_save, resized_img.cpu().data.numpy()[0][0]/255)
			else:
				return predicted_final_label

	def test_one_image_r(self, path, out_path):

		try:

			if self.config['grayscale']:
				image = self.profiler(self.test_data_loader.loader, path).convert('L')

			else:
				image = self.profiler(self.test_data_loader.loader, path).convert('RGB')

			self.test_one_image_with_cropped(image, out_path)

		except KeyboardInterrupt:

			self.profiler(log.info, 'Testing Interrupted')

			return False

			#Tests one image at a time,
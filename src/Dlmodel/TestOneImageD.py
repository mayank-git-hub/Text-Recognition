from .Dlmodel import Dlmodel
from ..helper.logger import Logger
from ..helper.utils import get_connected_components
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from torch.nn import functional as F
log = Logger()

class TestOneImageDClass(Dlmodel):

	def __init__(self, config, mode = 'train', profiler=None):

		print(config['project'])

		super().__init__(config=config, mode=mode, profiler=profiler)
		self.seed()

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def put_on_gpu(self, image):

		return [image.cuda()]

	def to_cpu(self, output):

		predicted_link = output[0][0, 2:, :, :].data.cpu().numpy().transpose(1, 2, 0)
		predicted_target = output[0][0, 0:2, :, :].data.cpu().numpy().transpose(1, 2, 0)

		return predicted_link, predicted_target

	def test_one_image_d(self, path, out_path):

		self.profiler(self.start_testing)

		with torch.no_grad():

			image_ = self.test_data_loader.loader(path)

			type_ = ['0.75', '1.25', '1'] # Always keep 1 at last
			base_r, base_c = 512 + 64*2, 768 + 64*2

			image_ = np.array(image_)
			original_image_shape = image_.shape

			if self.config['link']:
				output = [torch.FloatTensor(np.zeros([1, 8*2+2, base_r, base_c]))]
			else:
				output = [torch.FloatTensor(np.zeros([1, 2, base_r, base_c]))]

			if self.cuda:
				for i_ in range(len(output)):
					output[i_] = output[i_].cuda()

			for no_, type_i in enumerate(type_):

				row, column = int(base_r*float(type_i)), int(base_c*float(type_i))

				image, _ = self.test_data_loader._row_column_aspect_resize(row, column, image_.copy())

				image = self.test_data_loader.transform(image).unsqueeze(0)
				
				if self.cuda:
					image = self.put_on_gpu(image)
				
				temp_output = self.model(image)

				if self.cuda:
					output = [output[i_] + F.interpolate(temp_output[i_], size=(base_r, base_c)) for i_ in range(len(temp_output))]
				else:
					output = [output[i_] + F.interpolate(temp_output[i_].cpu(), size=(base_r, base_c)) for i_ in range(len(temp_output))]

			for i in range(len(output)):
				output[i_] /= len(type_)

			predicted_link, segmentation_predicted = self.to_cpu(output)
			resize = {'function': self.test_data_loader._row_column_inverse_resized_contours, 'base_r': base_r, 'base_c': base_c, 'original_image_shape': original_image_shape}
			contours = get_connected_components(segmentation_predicted, image_.astype(np.uint8), None, self.config, 'OWN', True, out_path, predicted_link, resize=resize)

			return contours

			#Tests one image at a time,

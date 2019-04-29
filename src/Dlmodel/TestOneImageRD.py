from .Dlmodel import Dlmodel
from ..helper.logger import Logger
from ..helper.utils import get_connected_components
import torch
import time
import random
import numpy as np
import editdistance
from .TestOneImageR import TestOneImageRClass
from .TestOneImageD import TestOneImageDClass
import matplotlib.pyplot as plt
from PIL import Image
log = Logger()

class TestOneImageRDClass():

	def __init__(self, config_r, config_d, mode = 'test_one', profiler=None):

		self.profiler = profiler
		self.cuda = True
		self.one_det = TestOneImageDClass(config_d, mode = 'test_one', profiler=profiler)
		self.one_rec = TestOneImageRClass(config_r, mode = 'test_one', profiler=profiler)

	def test_one_image_rd(self, path, out_path):

		try:

			image = np.array(Image.open(path).convert('RGB'), np.uint8)

			if len(image.shape) == 3:
				image = image[:, :, 0:3]
			else:
				image = image[:, :, None]
				image = np.repeat(image, 3, axis=2)

			with torch.no_grad():

				contours = self.one_det.test_one_image_d(path, out_path)

				all_cropped = []

				for cnt in contours:

					rotated = [self.one_rec.test_data_loader.rotate(cnt, image)]
					temp = self.one_rec.test_data_loader.resize(rotated)[0]
					if len(temp) == 0:
						print(cnt, temp, rotated[0].shape)
					all_cropped.append(temp)

				if self.cuda:
					# for all_ in all_cropped:
						# print(all_)
					all_cropped = [all_[0].cuda() for all_ in all_cropped]

				self.one_rec.model.eval()
				out = ['' for all_ in all_cropped]#self.one_rec.model(all_, decode=True)[1][0]

				return contours, out

		except KeyboardInterrupt:

			self.profiler(log.info, 'Testing Interrupted')

			return False

			#Tests one image at a time,
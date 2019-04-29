import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torchvision.transforms import RandomRotation
from PIL import Image

class trainLoader():

	def __init__(self, config=None, batch_size=50, epoch_len = 100, seq_len = 8, transform=None, training_path = '/home/mayank/Desktop/GitRepos/crnn-pytorch_mnist/data/processed/training.pt', ip_channels = 1, size=(20,200), Type='train', profiler = None, target_transform = None):

		self.training_path = training_path
		self.abc = '0123456789'
		self.seq_len = seq_len
		self.epoch_len = epoch_len
		self.transform = transform

		self.train_data, self.train_labels = torch.load(training_path)
		self.num_total = len(self.train_labels)
		self.final_size = size
		self.normal_mean = 7
		self.clip = (1,40)
		self.ip_channels = ip_channels
		self.resized_shape = (*size,ip_channels)

		self.target_aspect_ratio = 10

		self.out_dir = 'out'
		self.rotate = RandomRotation(10)

		self.batch_size = batch_size
		self.encoding_to_char = {1: '0', 2:'1', 3:'2', 4:'3', 5:'4', 6:'5', 7:'6', 8:'7', 9:'8', 10:'9'}

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def _update_config(self, config):

		self.config = config

	def get_abc(self):
		return self.abc

	def set_mode(self, mode='train'):
		return

	def generate_string(self):
		return ''.join(random.choice(self.abc) for _ in range(self.seq_len))


	def __len__(self):

		return self.epoch_len


	def instance(self, num):

		choices = np.random.randint(0, self.num_total, num)
		sample = self.train_data[choices]

		# Variable Spacing

		final = np.zeros([28, 28*num])

		if self.training_path == 'data/processed/training.pt':
			final[:, 0:28] = self.rotate(Image.fromarray(sample[0].numpy()).convert('L'))
		else:
			final[:, 0:28] = sample[0]
		till_now = 28

		for i in sample[1:]:

			
			i = Image.fromarray(i.numpy()).convert('L')
			if self.training_path == 'data/processed/training.pt':
				CENTER_DIST = 27 - np.random.randint(0, 14)
				rotated = self.rotate(i)
			else:
				CENTER_DIST = 27
				rotated = i
			
			final[:, till_now-(28 - CENTER_DIST):till_now-(28 - CENTER_DIST)+28] += rotated
			final[:, till_now-(28 - CENTER_DIST):till_now-(28 - CENTER_DIST)+28][final[:, till_now-(28 - CENTER_DIST):till_now-(28 - CENTER_DIST)+28]>255] = 255
			till_now += CENTER_DIST

		final = final[:, :till_now]

		labels = self.train_labels[choices]+1

		return final.astype(np.uint8), labels

	def batch(self, size):

		# http://www.ravi.io/language-word-lengths

		sizes = np.array([0.1, 0.6, 2.6, 5.2, 8.5, 12.2, 14, 14, 12.6, 10.1, 7.5, 5.2, 3.2, 2, 1, 0.6, 0.3, 0.2, 0.1, 0.1])
		sizes = sizes/np.sum(sizes)
		sizes = np.random.choice(np.arange(2, len(sizes)+2), size=size, p=sizes)

		images, targets = [], []
		for size in sizes:
			sample, target = self.instance(size)
			images.append(sample)
			targets.append(target)

		return images, targets

	def resize(self, to_resize_batch):

		#to_resize_batch: (examples, width, height, channels) 
		#upsample requires (batch_size, input_features, width, height)
		
		final = np.zeros([len(to_resize_batch), self.ip_channels, self.resized_shape[0], self.resized_shape[1]])

		for i, sample in enumerate(to_resize_batch):
			# print(sample.shape)
			c_aspect_ratio = sample.shape[1]//sample.shape[0]
			# print(c_aspect_ratio)
			
			if c_aspect_ratio > self.target_aspect_ratio:
				#Add along height
				new_h = int(sample.shape[0]*c_aspect_ratio/self.target_aspect_ratio)
				temp = np.zeros((new_h, sample.shape[1]))
				temp[(new_h-sample.shape[0])//2:(new_h-sample.shape[0])//2+sample.shape[0], :] = sample

			elif c_aspect_ratio < self.target_aspect_ratio:
				#Add along width

				new_w = int(sample.shape[1]*self.target_aspect_ratio/c_aspect_ratio)
				temp = np.zeros((sample.shape[0], new_w))
				temp[:, (new_w-sample.shape[1])//2:(new_w-sample.shape[1])//2+sample.shape[1]] = sample
			else:
				temp = sample

			sample = temp[None, None, :, :]
			final[i] = nn.functional.interpolate(torch.FloatTensor(sample), size=(self.resized_shape[0], self.resized_shape[1]), mode='nearest', align_corners=None)


		return final

	def getitem(self, index):

		size = self.batch_size
		images, targets = self.batch(size)
		images = [torch.FloatTensor(np.array(Image.fromarray(img).convert('RGB').resize(size=(int(32*img.shape[1]/img.shape[0]), 32))).transpose(2, 0, 1)) for img in images]

		seq_len = [torch.IntTensor([target_i.shape[0]]) for target_i in targets]

		seq = []
		for i in range(len(targets)):

			seq.append(targets[i].int())

		images = [image.unsqueeze(0) for image in images]

		sample = {"img": images, "seq": seq, "seq_len": seq_len, "aug": True}


		return sample

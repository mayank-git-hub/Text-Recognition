import torchvision
from .u_net_resnet_50_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .generic_model import model
from src.helper.logger import Logger

log = Logger()

import random

class UNetWithResnet50Encoder(model):

	def __init__(self, config, mode, profiler):
		
		super().__init__()

		self.config = config

		if not self.config['hard_negative_mining']:
			self.class_weight = torch.FloatTensor(self.config['class_weight']).cuda()
			
		self.seed()
		self.profiler = profiler
		self.DEPTH = 5 #3, 4, 5
		if config['lossf'] == 'CEL':
			if self.config['link']:
				self.classes = 18
			else:
				self.classes = 2

		self.loss_name = self.config['lossf']

		self.channel_depth = [64, 256, 512, 1024, 2048, 1024, 512, 256, 64, self.classes]

		profiler(self.define_architecture, profiler_type="once")
		
		self.prev_lr = config['lr'][1]

		if mode == 'train':

			if self.config['optimizer'] == 'Adam':
				log.info('Using Adam optimizer')
				self.opt = optim.Adam(self.parameters(), lr=config['lr'][1], weight_decay=config['weight_decay'])
			
			elif self.config['optimizer'] == 'SGD':
				log.info('Using SGD optimizer')
				self.opt = optim.SGD(self.parameters(), lr=config['lr'][1], momentum=config['momentum'], weight_decay=config['weight_decay'])

		if config['lossf'] == 'CEL':
			log.info('Using CEL')
			self.lossf = nn.CrossEntropyLoss(reduction='none')

	def define_architecture(self):

		resnet = torchvision.models.resnet.resnet50(pretrained=self.config['PreTrained_net'])
		
		self.input_block = nn.Sequential(*list(resnet.children()))[:3]
		self.input_pool = list(resnet.children())[3]
		down_blocks = []
		for bottleneck in list(resnet.children()):
			if isinstance(bottleneck, nn.Sequential):
				down_blocks.append(bottleneck)
		self.down_blocks = nn.ModuleList(down_blocks[0:self.DEPTH+1])
		del down_blocks

		self.bridge = Bridge(self.channel_depth[self.DEPTH - 1], self.channel_depth[9-self.DEPTH])

		up_blocks = []
		up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
		up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
		up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
		up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
													up_conv_in_channels=256, up_conv_out_channels=128))
		up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
													up_conv_in_channels=128, up_conv_out_channels=64))
		self.up_blocks = nn.ModuleList(up_blocks[5-self.DEPTH:])
		del up_blocks
		
		self.out = nn.Conv2d(64, self.classes, kernel_size=1, stride=1)
		self.output = nn.Softmax(dim=1)

		#UNetResNet architecture constructor

	def forward(self, x_big, with_output_feature_map=False):

		output_list = []
		if with_output_feature_map:
			output_feature_map_list = []

		for no, x in enumerate(x_big):

			pre_pools = dict()
			pre_pools["layer_0"] = x
			x = self.input_block(x)
			pre_pools["layer_1"] = x
			x = self.input_pool(x)
			x = self.down_blocks[0](x)
			pre_pools["layer_"+str(2)] = x

			for i, block in enumerate(self.down_blocks[1:], 3):
				x = block(x)
				
				if i == (self.DEPTH):
					break
				pre_pools["layer_"+str(i)] = x

			x = self.bridge(x)

			for i, block in enumerate(self.up_blocks, 1):
				
				key = "layer_"+str(self.DEPTH - i)
				x = block(x, pre_pools[key])

			output_feature_map = x
			x = self.out(x)
			del pre_pools

			if with_output_feature_map:
				output_list.append(x), output_feature_map_list.append(output_feature_map)
			else:
				output_list.append(x)

		if with_output_feature_map:
			return output_list, output_feature_map_list
		else:
			return output_list

	def __name__(self):

		return 'ResNet_UNet_forward_pass'

		#Returns string 'ResNet_UNet_forward_Pass' when called

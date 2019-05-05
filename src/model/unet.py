import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .unet_parts import *
import torchvision
from .generic_model import model
from src.helper.logger import Logger
log = Logger()
import random


class UNet(model):

	def __init__(self, config, profiler):
		
		super(UNet, self).__init__()

		super().__init__()

		self.config = config
			
		self.seed()
		self.profiler = profiler
		self.DEPTH = 5 #3, 4, 5
		if config['lossf'] == 'CEL':
			if self.config['link']:
				self.classes = 18
			else:
				self.classes = 2

		self.loss_name = self.config['lossf']

		self.inc = Inconv(self.config['n_channels'], 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		self.down4 = Down(512, 512)
		self.up1 = Up(1024, 256)
		self.up2 = Up(512, 128)
		self.up3 = Up(256, 64)#+128
		self.up4 = Up(128, 64)
		self.sigma = nn.Softmax(dim=-1)
		self.outc = OutConv(64, self.classes)


		self.prev_lr = config['lr'][1]

		if self.config['optimizer'] == 'Adam':
			log.info('Using Adam optimizer')
			self.opt = optim.Adam(self.parameters(), lr=config['lr'][1], weight_decay=config['weight_decay'])
		
		elif self.config['optimizer'] == 'SGD':
			log.info('Using SGD optimizer')
			self.opt = optim.SGD(self.parameters(), lr=config['lr'][1], momentum=config['momentum'], weight_decay=config['weight_decay'])

		if config['lossf'] == 'CEL':
			log.info('Using CEL')
			self.lossf = nn.CrossEntropyLoss(reduction='none')

	def forward(self, x_big):

		out_big = []

		for x in x_big:
			x1 = self.inc(x)
			x2 = self.down1(x1)
			x3 = self.down2(x2)
			x4 = self.down3(x3)
			x5 = self.down4(x4)
			x = self.up1(x5, x4)
			x = self.up2(x, x3)
			x = self.up3(x, x2)
			x = self.up4(x, x1)
			out_big.append(self.outc(x))

		return out_big
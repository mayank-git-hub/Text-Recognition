import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
import torchvision
from torchsummary import summary
import resnet_own

def own():
	nc=3
	rnn_hidden_size=256
	rnn_num_layers=2
	leakyRelu=False

	ks = [3, 3, 3, 3, 3, 3, 2]
	ps = [1, 1, 1, 1, 1, 1, 0]
	ss = [1, 1, 1, 1, 1, 1, 1]
	nm = [64, 128, 256, 256, 512, 512, 512]

	cnn = nn.Sequential()

	def convRelu(i, batchNormalization=False):
		nIn = nc if i == 0 else nm[i - 1]
		nOut = nm[i]
		cnn.add_module('conv{0}'.format(i),
					   nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
		if batchNormalization:
			cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
		if leakyRelu:
			cnn.add_module('relu{0}'.format(i),
						   nn.LeakyReLU(0.2, inplace=True))
		else:
			cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

	convRelu(0)
	cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
	convRelu(1)
	cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
	convRelu(2, True)
	convRelu(3)
	cnn.add_module('pooling{0}'.format(2),
				   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
	convRelu(4, True)
	convRelu(5)
	cnn.add_module('pooling{0}'.format(3),
				   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
	convRelu(6, True)  # 512x1x16

	print(summary(cnn.cuda(), (3,32,150)))

print("\n***OWN***\n")
own()
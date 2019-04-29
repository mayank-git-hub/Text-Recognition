import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.models as models
import string
import numpy as np
from warpctc_pytorch import CTCLoss
from .generic_model import model
from ..helper.logger import Logger
import torch.optim as optim
import torchvision

log = Logger()


class BidirectionalLSTM(nn.Module):

	def __init__(self, nIn, nHidden, nOut):
		super(BidirectionalLSTM, self).__init__()

		self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
		self.embedding = nn.Linear(nHidden * 2, nOut)

	def forward(self, input):
		recurrent, _ = self.rnn(input)
		T, b, h = recurrent.size()
		t_rec = recurrent.view(T * b, h)

		output = self.embedding(t_rec)  # [T * b, nOut]
		output = output.view(T, b, -1)

		return output


class CRNN_orig(model):

	def __init__(self, abc, config, profiler, imgH, nc, rnn_hidden_size=256, rnn_num_layers=2, leakyRelu=False):
		
		super(CRNN_orig, self).__init__()
		assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

		self.abc = abc
		self.num_classes = len(abc)+1
		self.hidden_size = rnn_hidden_size

		self.config = config
		self.seed()
		self.profiler = profiler

		self.loss_name = self.config['lossf']
		
		self.prev_lr = config['lr'][1]
		self.is_gray = (nc == 1)

		self.deep = False

		if not self.deep:

			ks = [3, 3, 3, 3, 3, 3, 2]
			ps = [1, 1, 1, 1, 1, 1, 0]
			ss = [1, 1, 1, 1, 1, 1, 1]
			nm = [64, 128, 256, 256, 512, 512, 512]

			cnn = nn.Sequential()

			def convRelu(i, batchNormalization=True):
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

			self.cnn = cnn
			self.rnn = nn.Sequential(
				BidirectionalLSTM(512, self.hidden_size, self.hidden_size),
				BidirectionalLSTM(self.hidden_size, self.hidden_size, self.num_classes))

		else:

			ks = [3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
			ps = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
			ss = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
			nm = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512]

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

			convRelu(0, True)
			convRelu(1, True)
			cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64

			convRelu(2, True)
			convRelu(3, True)
			cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32

			convRelu(4, True)
			convRelu(5, True)
			cnn.add_module('pooling{0}'.format(2),
						   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
			
			convRelu(6, True)
			convRelu(7, True)
			cnn.add_module('pooling{0}'.format(3),
						   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16

			convRelu(8, True)
			convRelu(9, True)
			cnn.add_module('pooling{0}'.format(3),
						   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))			

			self.cnn = cnn
			self.rnn = nn.Sequential(
				BidirectionalLSTM(512, self.hidden_size, self.hidden_size),
				BidirectionalLSTM(self.hidden_size, self.hidden_size, self.num_classes))


		if config['lossf'] == 'CTC':
			log.info('Using CTC')
			self.lossf = CTCLoss()

		if self.config['optimizer'] == 'Adam':
			log.info('Using Adam optimizer')
			self.opt = optim.Adam(self.parameters(), lr=config['lr'][1], weight_decay=config['weight_decay'])
		
		elif self.config['optimizer'] == 'SGD':
			log.info('Using SGD optimizer')
			self.opt = optim.SGD(self.parameters(), lr=config['lr'][1], momentum=config['momentum'], weight_decay=config['weight_decay'])

		if self.config['PreTrained_net'] and self.config['PreTrained_model']['check'] == False:
			self.custom_pick()

		if config['varying_width']:
			self.list_or_tensor = 'list'
		else:
			self.list_or_tensor = 'tensor'
			

	def forward(self, input):

		conv = self.cnn(input)
		b, c, h, w = conv.size()
		assert h == 1, "the height of conv must be 1"
		conv = conv.squeeze(2)
		conv = conv.permute(2, 0, 1)  # [w, b, c]
		output = self.rnn(conv)

		return output

	def loss(self, preds, labels, pred_lens, label_lens, info):

		if self.list_or_tensor == 'tensor':

			loss = self.lossf(preds, labels, pred_lens, label_lens)/pred_lens.shape[0]

		else:
			loss = 0

			for i in range(len(preds)):
				loss += self.lossf(preds[i], labels[i], pred_lens[i], label_lens[i])

			loss /= len(preds)

		if info['Keep_log']:

			info['Loss'].append(loss.data.cpu().numpy()[0])

		else:

			info['Loss'] = (info['Loss']*info['Count'] + loss.data.cpu().numpy()[0])/(info['Count']+1)

		info['Count'] += 1
		info['Loss'].append(loss.data.cpu().numpy()[0])

		return loss

	def print_info(self, info, iterator, status=''):
		status = "Loss: {0:.6f}; Avg Loss: {1:.6f}".format(info['Loss'][-1], np.mean(np.array(info['Loss'][-min(200, len(info['Loss'])):])))
		iterator.set_description(status)

	def custom_pick(self):
		
		if self.is_gray:
			path = 'crnn.pth'
		else:
			path = 'crnn3.pth'

		model_path = self.config['PreTrained_path']+path
		log.info('Loading pretrained model from %s' % model_path)
		orig_dict = torch.load(model_path)

		if not self.deep: #original non-deep model

			if len(self.abc) != 37:
				# 1. filter out unnecessary keys
				pretrained_dict = {k: v for k, v in orig_dict.items() if k not in ['rnn.1.embedding.weight', 'rnn.1.embedding.bias']}

				weights = torch.ones(len(self.abc)+1,512)*torch.mean(orig_dict['rnn.1.embedding.weight'],0)
				biases = torch.ones(len(self.abc)+1)*torch.mean(orig_dict['rnn.1.embedding.bias'])

				weights[:37,:] = orig_dict['rnn.1.embedding.weight']
				biases[:37] = orig_dict['rnn.1.embedding.bias']
				pretrained_dict['rnn.1.embedding.weight'] = weights
				pretrained_dict['rnn.1.embedding.bias'] = biases

				# 2. overwrite entries in the existing state dict
				self.state_dict().update(pretrained_dict) 
				# 3. load the new state dict
				self.load_state_dict(pretrained_dict)
			else:
				self.load_state_dict(orig_dict)

		else: #new model
			
			pretrained_dict = {k: v for k, v in orig_dict.items() if 'cnn' not in k}

			if len(self.abc) != 37:
				pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['rnn.1.embedding.weight', 'rnn.1.embedding.bias']}
				weights = torch.ones(len(self.abc)+1,512)*torch.mean(orig_dict['rnn.1.embedding.weight'],0)
				biases = torch.ones(len(self.abc)+1)*torch.mean(orig_dict['rnn.1.embedding.bias'])

				weights[:37,:] = orig_dict['rnn.1.embedding.weight']
				biases[:37] = orig_dict['rnn.1.embedding.bias']
				pretrained_dict['rnn.1.embedding.weight'] = weights
				pretrained_dict['rnn.1.embedding.bias'] = biases

			self.load_state_dict(pretrained_dict, strict=False)


class CRNN_resnet(model):

	def __init__(self, abc, config, profiler, imgH, nc, rnn_hidden_size=256, rnn_num_layers=2, leakyRelu=False):
		
		super(CRNN_resnet, self).__init__()
		assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

		self.abc = abc
		self.num_classes = len(abc)+1
		self.hidden_size = rnn_hidden_size

		self.use_pretrained_rnn = False

		self.config = config
		self.seed()
		self.profiler = profiler

		self.loss_name = self.config['lossf']
		
		self.prev_lr = config['lr'][1]
		self.is_gray = (nc == 1)

		resnet = torchvision.models.resnet.resnet18(pretrained=True)

		for param in resnet.parameters():
		    param.requires_grad = False

		features = list(resnet.children())[:-2]
		self.cnn = nn.Sequential(
			*features,
			)#nn.Conv2d(1024, 512, 2)

		self.rnn = nn.Sequential(
			BidirectionalLSTM(512, self.hidden_size, self.hidden_size),
			BidirectionalLSTM(self.hidden_size, self.hidden_size, self.num_classes))

		if config['lossf'] == 'CTC':
			log.info('Using CTC')
			self.lossf = CTCLoss()

		if self.config['optimizer'] == 'Adam':
			log.info('Using Adam optimizer')
			self.opt = optim.Adam(self.parameters(), lr=config['lr'][1], weight_decay=config['weight_decay'])
		
		elif self.config['optimizer'] == 'SGD':
			log.info('Using SGD optimizer')
			self.opt = optim.SGD(self.parameters(), lr=config['lr'][1], momentum=config['momentum'], weight_decay=config['weight_decay'])

		if self.config['PreTrained_net'] and self.config['PreTrained_model']['check'] == False and self.use_pretrained_rnn:
			self.custom_pick()

		if config['varying_width']:
			self.list_or_tensor = 'list'
		else:
			self.list_or_tensor = 'tensor'
			

	def forward(self, input):

		conv = self.cnn(input)
		b, c, h, w = conv.size()
		assert h == 1, "the height of conv must be 1"
		conv = conv.squeeze(2)
		conv = conv.permute(2, 0, 1)  # [w, b, c]
		output = self.rnn(conv)

		return output

	def loss(self, preds, labels, pred_lens, label_lens, info):

		if self.list_or_tensor == 'tensor':

			loss = self.lossf(preds, labels, pred_lens, label_lens)/pred_lens.shape[0]

		else:
			loss = 0

			for i in range(len(preds)):
				loss += self.lossf(preds[i], labels[i], pred_lens[i], label_lens[i])

			loss /= len(pred_lens)

		if info['Keep_log']:

			info['Loss'].append(loss.data.cpu().numpy()[0])

		else:

			info['Loss'] = (info['Loss']*info['Count'] + loss.data.cpu().numpy()[0])/(info['Count']+1)

		info['Count'] += 1
		info['Loss'].append(loss.data.cpu().numpy()[0])

		return loss

	def print_info(self, info, iterator, status=''):
		status = "Loss: {0:.6f}; Avg Loss: {1:.6f}".format(info['Loss'][-1], np.mean(np.array(info['Loss'])))
		iterator.set_description(status)

	def custom_pick(self):
		
		if self.is_gray:
			path = 'crnn.pth'
		else:
			path = 'crnn3.pth'

		model_path = self.config['PreTrained_path']+path
		log.info('Loading pretrained model from %s' % model_path)
		orig_dict = torch.load(model_path)

		if len(self.abc) != 37:
			# 1. filter out unnecessary keys
			pretrained_dict = {k: v for k, v in orig_dict.items() if (k not in ['rnn.1.embedding.weight', 'rnn.1.embedding.bias'] and 'cnn' not in k)}

			weights = torch.ones(len(self.abc)+1,512)*torch.mean(orig_dict['rnn.1.embedding.weight'],0)
			biases = torch.ones(len(self.abc)+1)*torch.mean(orig_dict['rnn.1.embedding.bias'])

			weights[:37,:] = orig_dict['rnn.1.embedding.weight']
			biases[:37] = orig_dict['rnn.1.embedding.bias']
			pretrained_dict['rnn.1.embedding.weight'] = weights
			pretrained_dict['rnn.1.embedding.bias'] = biases

			# 2. overwrite entries in the existing state dict
			# self.state_dict().update(pretrained_dict) 
		else:
			pretrained_dict = {k: v for k, v in orig_dict.items() if 'cnn' not in k}			
		
		self.load_state_dict(pretrained_dict, strict=False)
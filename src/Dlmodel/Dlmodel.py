import sys
from torchvision import transforms
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from ..helper.read_yaml import read_yaml
from ..helper.logger import Logger

log = Logger()

class Dlmodel():

	def __init__(self, config, mode = 'train', profiler=None):

		# data_loader object created
		# Removes randomness
		# Creates dataloader object
		# Dictionaries created to record results of training, testing, and best model
		# loads previous model from checkpoint if required
		
		self.profiler = profiler
		self.config = config
		profiler(self.seed, profiler_type="once") #This removes randomness
		self.cuda = self.config['cuda'] and torch.cuda.is_available()
		self.mode = mode

		if self.config['project'] == 'Text_Detection':

			self.text_detection()

		elif self.config['project'] == 'Text_Recognition':

			self.text_reco()

		if mode == 'train' or mode == 'test':

			self.init_savers()

			if self.cuda:
				self.model.cuda() #if GPU is available, then use it

			self.start_no = 0

			if self.config['PreTrained_model']['check'] == True or mode=='testing':

				self.get_pretrained()

		elif mode == 'test_one':

			self.profiler(self.get_transforms, profiler_type="once")

			if self.cuda:
				self.profiler(self.model.cuda)

			if self.config['PreTrained_model']['while_testing']:

				self.model_best = self.profiler(torch.load, self.config['PreTrained_model']['checkpoint_best'])['best']
				self.training_info = self.profiler(self.model.load, self.config['PreTrained_model']['checkpoint_best'], self.config['PreTrained_model']['checkpoint_best_info'], mode, profiler_type="once")
				self.start_no = int(self.config['PreTrained_model']['checkpoint_best'].split('/')[-1].split('_')[0])

			log.info('Loaded the model')

	def get_model(self, model, profiler):

		return profiler(model, config=self.config, profiler=profiler, profiler_type="once")
		#Calls the funtion, with specified classes and channels

	def init_savers(self):

		if self.config['project'] == 'Text_Detection':

			self.plot_training = {'Loss' : [], 'Acc' : []}	
			self.plot_testing = {'Loss' : [], 'Acc' : []}

			self.training_info = {'Loss': [], 'Seg_loss':[], 'Link_Loss':[], 'Class_balanced_Loss':[], 'Reco_Loss':[], 'Acc': [], 'Keep_log': True, 'Count':0}
			self.testing_info = {'Acc': 0, 'Loss': 0, 'Seg_loss':0, 'Link_Loss':0, 'Class_balanced_Loss':0, 'Reco_Loss':0, 'Count': 0, 'Keep_log': False}

			self.model_best = {'Loss': sys.float_info.max, 'Acc': 0.0}

		elif self.config['project'] == 'Text_Recognition':

			self.plot_training = {'Loss' : []}	
			self.plot_testing = {'Acc' : []}

			self.training_info = {'Loss': [], 'Keep_log': True, 'Count':0}

			self.testing_info = {'Acc': 0, 'editdistance': 0}

			self.model_best = {'Acc': 0, 'editdistance': 100}

		# dictionaries to record the results of testing, training, and best model

	def get_pretrained(self):

		self.model_best = self.profiler(torch.load, self.config['PreTrained_model']['checkpoint_best'])['best']
		self.profiler(self.model.load, self.config['PreTrained_model']['checkpoint'], self.config['PreTrained_model']['checkpoint_info'], self.mode, profiler_type="once")

		self.start_no = int(self.config['PreTrained_model']['checkpoint'].split('/')[-1].split('_')[0])

		if self.mode == 'train':

			check = True and os.path.exists(self.config['dir']['Plots']+'/training_loss.npy')
			check = check and os.path.exists(self.config['dir']['Plots']+'/training_accuracy.npy')
			check = check and os.path.exists(self.config['dir']['Plots']+'/testing_accuracy.npy')
			check = check and os.path.exists(self.config['dir']['Plots']+'/testing_loss.npy')

			if check:

				self.plot_training['Loss'] = list(np.load(self.config['dir']['Plots']+'/training_loss.npy'))
				self.plot_training['Acc'] = list(np.load(self.config['dir']['Plots']+'/training_accuracy.npy'))
				self.plot_testing['Acc'] = list(np.load(self.config['dir']['Plots']+'/testing_accuracy.npy'))
				self.plot_testing['Loss'] = list(np.load(self.config['dir']['Plots']+'/testing_loss.npy'))
		
		self.profiler(log.info, 'Loaded the model')

				#if testing mode, or we want to use a pretrained model, loading model from previous
	def text_detection(self):

		if self.config['aspect_resize'] == 'scale_two':
			from ..loader.scale_two import scale_two as own_DataLoader
		elif self.config['aspect_resize'] == 'constant':
			from ..loader.square import square as own_DataLoader
		elif self.config['aspect_resize'] == '1-0.5-2':
			from ..loader.dete_loader import DeteLoader as own_DataLoader

		log.info('Using Model:', self.config['name'])
		if self.config['name'] == 'UNet':
			from ..model.unet import UNet
			self.model = self.profiler(self.get_model, UNet, self.profiler, profiler_type="once")
		elif self.config['name'] == 'UNet_Resnet_50':
			from ..model.u_net_resnet_50_encoder import UNetWithResnet50Encoder
			self.model = self.profiler(self.get_model, UNetWithResnet50Encoder, self.profiler, profiler_type="once")

		self.profiler(self.get_transforms, profiler_type="once")
		
		if self.mode != 'test_one':

			self.train_data_loader = self.profiler(own_DataLoader, self.config, type_='train', profiler = self.profiler, transform=self.train_transform, target_transform = self.target_transform, profiler_type="once")
			self.test_data_loader = self.profiler(own_DataLoader, self.config, type_='test', profiler = self.profiler, transform=self.test_transform, target_transform = self.target_transform, profiler_type="once")
		else:
			self.test_data_loader = self.profiler(own_DataLoader, self.config, type_='test_one', profiler = self.profiler, transform=self.test_transform, target_transform = self.target_transform, profiler_type="once")

	def text_reco(self):

		if self.config['loader'] == 'reco':

			from ..loader.reco_loader import RecoDataloader as own_DataLoader
		
		elif self.config['loader'] == 'MNIST':

			from ..loader.mnist import trainLoader as own_DataLoader

		log.info('Getting the Dataset')

		if self.config['grayscale']:
			self.channels = 1
		else:
			self.channels = 3

		self.profiler(self.get_transforms, profiler_type="once")
		channels = self.channels
		
		if self.mode != 'test_one':
			
			self.train_data_loader = self.profiler(own_DataLoader, self.config, 'train', channels, profiler = self.profiler, transform=self.train_transform, target_transform = self.target_transform, profiler_type="once")
			self.test_data_loader = self.profiler(own_DataLoader, self.config, 'test', channels, profiler = self.profiler, transform=self.test_transform, target_transform = self.target_transform, profiler_type="once")
		else:
			self.test_data_loader = self.profiler(own_DataLoader, self.config, 'test_one', channels, profiler = self.profiler, transform=self.test_transform, target_transform = self.target_transform, profiler_type="once")

		if self.config['conv'] == 'resnet':
			from ..model.crnn import CRNN_resnet as CRNN
		elif self.config['conv'] == 'orig':
			from ..model.crnn import CRNN_orig as CRNN
		else:
			print("reco in config should be either resnet or orig")
			exit(0)

		#Loading the CRNN model. Load weights if provided, otherwise from scratch
		#Seq proj is an additional conv layer where 10 and 20 are input and ouput channels
		self.model = self.profiler(CRNN, self.test_data_loader.get_abc(), self.config, self.profiler, self.config['img_h'], channels, profiler_type="once")

	def get_transforms(self):

		if self.config['train']['transform'] == True:
			self.train_transform = transforms.Compose([
											transforms.ColorJitter(brightness=self.config['augmentation']['brightness'], contrast=self.config['augmentation']['contrast'], saturation=self.config['augmentation']['saturation'], hue=self.config['augmentation']['hue']),
											transforms.ToTensor(),
											])
		else:
			self.train_transform = transforms.Compose([
											transforms.ToTensor(),
											])

		self.test_transform = transforms.Compose([
										transforms.ToTensor(),
										])

		self.target_transform = transforms.Compose([
										transforms.ToTensor(),
										])

		#Does data augmentation, ie. tranforms images by changing colour, hue brightness, etc., and returns tensor

	def get_config(self, path='configs/config.yaml'):

		return read_yaml(path)
		#To load hyperparameters from yaml file, dictionary format

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True
		#This removes randomness, makes everything deterministic

	def convert_argmax_to_channels(self, temp, masks):

		t_show = np.zeros([temp.shape[0], temp.shape[1], masks]).astype(np.uint8)

		#t_show is numpy array with rows=temp.shape[0], columns=temp.shape[1] and number of channels=masks

		for __i in range(t_show.shape[0]):
			for __j in range(t_show.shape[1]):
				t_show[__i, __j, temp[__i, __j]] = 255

		return t_show
		#returns numpy array with rows=temp.shape[0], columns=temp.shape[1] and channels=masks, with temp[_i,_j] channel of __i,__j element set to 255, rest are 0

	def start_training(self):

		self.model.requires_grad = True

		self.model.train()

		self.model.opt.zero_grad()

		#While training, gradients are required, gradients are set to zero to prevent accumulation of gradients with each pass

	def start_testing(self):

		self.model.requires_grad = False

		self.model.eval()

		#Gradient calculation not required as no backprop

	def show_graph(self):

		plt.clf()

		plt.subplot(211)
		if 'Loss' in self.plot_training:
			plt.plot(self.plot_training['Loss'], color='red')
		if 'Loss' in self.plot_testing:
			plt.plot(self.plot_testing['Loss'], color='blue')
		plt.title('Upper Plot: Loss, Red:train, Blue:Testing\nLower Plot: Accuracy, Red:train, Blue:Testing')
		plt.subplot(212)
		if 'Acc' in self.plot_training:
			plt.plot(self.plot_training['Acc'], color='red')
		if 'Acc' in self.plot_testing:
			plt.plot(self.plot_testing['Acc'], color='blue')
		plt.savefig(self.config['dir']['Plots']+'/Plot.png')
		plt.clf()

		#Plots the training and testing accuracy and loss, and saves it

	def test_module(self, func):

		torch.cuda.synchronize()
		torch.cuda.empty_cache()

		func()

		torch.cuda.synchronize()
		torch.cuda.empty_cache()

		if 'Loss' in self.plot_training:
			self.plot_training['Loss'].append(np.mean(self.training_info['Loss']))
		if 'Acc' in self.plot_training:
			self.plot_training['Acc'].append(np.mean(self.training_info['Acc']))

		self.start_training()

		if 'Acc' in self.plot_testing:
			np.save(self.config['dir']['Plots']+'/testing_accuracy.npy', self.plot_testing['Acc'])
		if 'Loss' in self.plot_testing:
			np.save(self.config['dir']['Plots']+'/testing_loss.npy', self.plot_testing['Loss'])
		if 'Acc' in self.plot_training:
			np.save(self.config['dir']['Plots']+'/training_accuracy.npy', self.plot_training['Acc'])
		if 'Loss' in self.plot_training:
			np.save(self.config['dir']['Plots']+'/training_loss.npy', self.plot_training['Loss'])

	def put_on_cuda(self, *args):

		return [[i.cuda() for i in args_i] for args_i in args if args_i is not None]

	def to_cpu(self, output, data, target):
		
		predicted_target = output[0][0, 0:2, :, :].data.cpu().numpy().transpose(1, 2, 0)
		numpy_data = data[0][0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 0:3]
		numpy_target = target[0][0, 0].data.cpu().numpy()
		if self.config['link']:
			predicted_link = output[0][0, 2:, :, :].data.cpu().numpy().transpose(1, 2, 0)
			return predicted_link, predicted_target, numpy_data, numpy_target

		return predicted_target, numpy_data, numpy_target

	def __name__(self):

		return 'Dlmodel'

	def __str__(self):

		return str(self.config)

		#returns string respresentation of object
from .Dlmodel import Dlmodel
from ..helper.logger import Logger
from ..helper.utils import get_connected_components
import numpy as np
import torch
from torch.nn import functional as F
import random
from tqdm import tqdm

log = Logger()

class TrainTestD(Dlmodel):
	
	def __init__(self, config, mode = 'train', profiler=None):

		profiler(super().__init__, config, mode, profiler)
		self.seed()

	def seed(self):

		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def check_show_f(self, d_name, path, contour, data, target, link):

		import matplotlib.pyplot as plt
		import cv2

		print(d_name[0], path[0], contour[0].shape)

		show_t = np.zeros([data[0].shape[2], data[0].shape[3], 3])
		show_t[:, :, 0] = target[0][0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 0]

		show_l = np.zeros([data[0].shape[2], data[0].shape[3], 3])
		show_l[:, :, 0] = link[0][0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 0]
		image = (data[0][0].data.cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8).copy()
		cv2.drawContours(image, np.array(contour[0]).astype(np.int32), -1, (0, 255, 0), 3)

		plt.imshow(np.concatenate((data[0][0].data.cpu().numpy().transpose(1, 2, 0), show_t, show_l), axis=1))
		plt.show()
		plt.imshow(data[0][0].data.cpu().numpy().transpose(1, 2, 0)*show_t)
		plt.show()
		plt.imshow(image)
		plt.show()

	def check_cpu_f(self, data):

		self.model = self.model.cpu()

		checking = time.time()

		for i in range(5):
			print(i)
			with torch.no_grad():
				self.profiler(self.model, [data[0]])

		print((time.time() - checking)/5)

	def calc_loss(self, data_list, info):

		data, output, target, contour, d_name, link, weight = data_list

		if self.config['weight_bbox'] and self.config['link']:

			loss = self.profiler(self.model.loss, [i.data.cpu().numpy() for i in data], output, target, contour, info, d_name, link_big=link, weight_big=weight)

		elif self.config['weight_bbox']:

			loss = self.profiler(self.model.loss, [i.data.cpu().numpy() for i in data], output, target, contour, info, d_name, link_big=None, weight_big=weight)

		elif self.config['link']:

			loss = self.profiler(self.model.loss, [i.data.cpu().numpy() for i in data], output, target, contour, info, d_name, link_big=link, weight_big=None)

		else:

			loss = self.profiler(self.model.loss, [i.data.cpu().numpy() for i in data], output, target, contour, info, d_name, link_big=None, weight_big=None)

		return loss

	def periodic_checks(self, no, output, data, target, d_name):

		if no%self.config['update_config']==0 and no!=0:

			prev_config = self.config

			self.config = self.profiler(self.get_config) #We can make changes to yaml file in real time
			self.train_data_loader._update_config(self.config)
			self.test_data_loader._update_config(self.config)
			self.model._update_config(self.config)

			if self.config['lr']!=prev_config['lr']:
				self.profiler(self.model.dynamic_lr, self.config['lr'], no)
				self.profiler(log.info, 'Learning Rate Changed from ', prev_config['lr'], ' to ', self.config['lr'])

		if no%self.config['log_interval_steps'] == 0 and no!=0:

			self.profiler(self.model.save, no=no, info = self.training_info, best=self.model_best, is_best = False)

		if no%self.config['show_some']==0 and no!=0:

			if self.config['link']:
				predicted_link, predicted_target, numpy_data, numpy_target = self.profiler(self.to_cpu, output, data, target)
				self.profiler(get_connected_components, predicted_target, numpy_data, numpy_target, self.config, d_name[0], True, link_pred=predicted_link)
			else:
				predicted_target, numpy_data, numpy_target = self.profiler(self.to_cpu, output, data, target)
				self.profiler(get_connected_components, predicted_target, numpy_data, numpy_target, self.config, d_name[0], True)


	def train_d(self):

		iterator = tqdm(range(self.start_no, self.config['steps']))

		check_show = False
		check_cpu = False

		for no in iterator:

			(data, link, target, weight, contour, path, text, d_name) = self.profiler(self.train_data_loader.getitem, no)
			if self.cuda:

				data, target, link, weight = self.profiler(self.put_on_cuda, data, target, link, weight)

			self.profiler(self.model.update_lr, no)

			if check_show:

				self.check_show_f(d_name, path, contour, data, target, link)

			if check_cpu:

				self.check_cpu_f(data)

				return True

			#if GPU available, use it			

			output = self.profiler(self.model, data)

			self.periodic_checks(no, output, data, target, d_name)
			loss = self.calc_loss([data, output, target, contour, d_name, link, weight], self.training_info)

			self.profiler(loss.backward)
			self.profiler(self.model.opt.step)
			self.profiler(self.model.opt.zero_grad)
			self.profiler(self.model.print_info, self.training_info, iterator)

			if self.config['link']:
				del link

			if self.config['weight_bbox']:
				del weight

			del data, target, output, loss, contour, path

			if no%self.config['test_now']==0 and no!=0:

				self.profiler(self.test_module, self.test_d)
			
			self.profiler.step()

		self.profiler.dump()

		self.start_no = 0
		self.training_info = {'Loss': [], 'Seg_loss':[], 'Link_Loss':[], 'Class_balanced_Loss':[], 'Reco_Loss':[], 'Acc': [], 'Keep_log': True, 'Count':0}


		# Performs forward propogation and backward propogation, updates learning rate, can make changes in yaml file in real time
		# At regular intervals, model is tested(validation) and the corresponding testing accuracy is plotted
		# Similarly training accuracy also plotted
		# Saves the model regularly

	def test_d(self):

		# Tests model from testing data, saves best model if accuracy exceeds previous best
		# Saves model at regualar intervals
		# Records the time taken for model to output per image

		self.profiler(self.start_testing)

		with torch.no_grad():

			iterator = tqdm(range(len(self.test_data_loader)))

			type_ = ['0.75', '1.25', '1'] # Always keep 1 at last
			base_r, base_c = 512 + 64*2, 768 + 64*2

			check_show = False

			for no in iterator:

				if self.config['link']:
					output = [torch.FloatTensor(np.zeros([1, 8*2+2, base_r, base_c])) for i in range(self.test_data_loader.batchsize)]
				else:
					output = [torch.FloatTensor(np.zeros([1, 2, base_r, base_c])) for i in range(self.test_data_loader.batchsize)]

				if self.cuda:
					for i_ in range(len(output)):
						output[i_] = output[i_].cuda()

				for no_, type_i in enumerate(type_):

					(data, link, target, weight, contour, path, text, d_name) = self.profiler(self.test_data_loader.getitem, no, type_i)
					if self.cuda:
						data, target, link, weight = self.profiler(self.put_on_cuda, data, target, link, weight)

					if check_show:

						import matplotlib.pyplot as plt

						print(d_name[0], path[0])

						show_t = np.zeros([data[0].shape[2], data[0].shape[3], 3])
						print(target[0][0].data.cpu().numpy().transpose(1, 2, 0).shape)
						show_t[:, :, 0] = target[0][0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 0]
						plt.imshow(np.concatenate((data[0][0].data.cpu().numpy().transpose(1, 2, 0), show_t), axis=1))
						plt.show()
						plt.imshow(data[0][0].data.cpu().numpy().transpose(1, 2, 0)*show_t)
						plt.show()
						
					temp_output = self.profiler(self.model, data)

					if self.cuda:
						output = [output[i_] + F.interpolate(temp_output[i_], size=(base_r, base_c)) for i_ in range(len(temp_output))]
					else:
						output = [output[i_] + F.interpolate(temp_output[i_].cpu(), size=(base_r, base_c)) for i_ in range(len(temp_output))]
				
				for i in range(len(output)):
					output[i_] /= len(type_)

				loss = self.calc_loss([data, output, target, contour, d_name, link, weight], self.testing_info)

				status = 'Testing: Best Acc: '+str(self.model_best['Acc'])+' Least Loss: '+str(self.model_best['Loss'])+' '
				self.profiler(self.model.print_info, self.testing_info, iterator, status)

				if no%self.config['save_test_image'] == 0 and no!=0:

					if self.config['link']:
						predicted_link, predicted_target, numpy_data, numpy_target = self.profiler(self.to_cpu, output, data, target)
						self.profiler(get_connected_components, predicted_target, numpy_data, numpy_target, self.config, d_name[0], True, path=self.config['dir']['Test_sample']+'/'+str(no), link_pred=predicted_link)
					else:
						predicted_target, numpy_data, numpy_target = self.profiler(self.to_cpu, output, data, target)
						self.profiler(get_connected_components, predicted_target, numpy_data, numpy_target, self.config, d_name[0], True, path=self.config['dir']['Test_sample']+'/'+str(no))

				if self.config['link']:
					del link

				if self.config['weight_bbox']:
					del weight

				del data, target, output, loss

		status_final = 'Testing: Best Acc: '+str(int(self.model_best['Acc']*10000)/10000)+' Least Loss: '+str(int(self.model_best['Loss']*10000)/10000)+' '

		if self.mode =='train':

			if self.testing_info['Acc'] > self.model_best['Acc'] or (self.testing_info['Acc'] == self.model_best['Acc'] and self.testing_info['Loss'] < self.model_best['Loss']):
				
				status_final += 'New best model found!! '
				self.model_best['Acc'] = self.testing_info['Acc']
				self.model_best['Loss'] = self.testing_info['Loss']
				
				self.profiler(self.model.save, no=0, info = self.testing_info, best=self.model_best, is_best=True)

			self.plot_testing['Acc'].append(self.testing_info['Acc'])
			self.plot_testing['Loss'].append(self.testing_info['Loss'])

		status_final += "Avg Loss: {}; Avg Accuracy: {}".format(self.testing_info['Loss'], self.testing_info['Acc'])

		iterator.set_description(status_final)

		self.testing_info = {'Acc': 0, 'Loss': 0, 'Seg_loss':0, 'Link_Loss':0, 'Class_balanced_Loss':0, 'Reco_Loss':0, 'Count': 0, 'Keep_log': False}

		return True
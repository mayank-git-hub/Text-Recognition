from .Dlmodel import Dlmodel
from ..helper.logger import Logger
from ..helper.utils import get_connected_components
import torch
import time
import random
import numpy as np
import editdistance
import tqdm as tqdm

log = Logger()

class TrainTestR(Dlmodel):
	
	def __init__(self, model, mode = 'train', profiler=None, target_transform=None, train_transform=None, test_transform=None):

		super().__init__(model, mode, profiler, target_transform, train_transform, test_transform)
		self.seed()

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def periodic_check(self, no):

		if no%self.config['update_config']==0 and no!=0:

			prev_config = self.config

			self.config = self.profiler(self.get_config, 'configs/text_config.yaml') #We can make changes to yaml file in real time
			self.train_data_loader._update_config(self.config)
			self.test_data_loader._update_config(self.config)
			self.model._update_config(self.config)

			if self.config['lr']!=prev_config['lr']:
				self.profiler(self.model.dynamic_lr, self.config['lr'], no)
				self.profiler(log.info, 'Learning Rate Changed from ', prev_config['lr'], ' to ', self.config['lr'])

		if no%self.config['print_log_steps'] == 0 and no!=0:# and (self.start_no + no) > 75:
			self.profiler(log.info)
			self.profiler(self.model.print_info(iterator), self.training_info)

		if no%self.config['log_interval_steps'] == 0 and no!=0:

			self.profiler(self.model.save, no=no, info = self.training_info, best=self.model_best, is_best = False)

	def check_cpu_f(self, imgs, sample):

		import matplotlib.pyplot as plt

		start = 0

		for j in range(imgs.shape[0]):
			print(''.join([self.test_data_loader.encoding_to_char[i] for i in sample["seq"][start:start+label_lens[j]].data.cpu().numpy()]))
			plt.imshow(imgs[j].data.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
			plt.show()
			start += label_lens[j]

	def train_r(self):

		self.profiler(self.start_training)

		iterator = tqdm(np.arange(self.config['steps']))
		check_cpu = False
		for no in iterator:

			sample = self.profiler(self.train_data_loader.getitem, no)

			self.profiler(self.model.update_lr, epoch_i*len(self.train_data_loader) + self.start_no + no+1)

			imgs = sample["img"]
			labels = sample["seq"].view(-1)
			label_lens = sample["seq_len"].int()

			if check_cpu:
				self.check_cpu(imgs, sample)

			#if GPU available, use it

			if self.cuda:

				imgs = imgs.cuda()	

			preds = self.profiler(self.model, imgs).cpu()
			pred_lens = torch.IntTensor([preds.size(0)] * imgs.shape[0])

			loss = self.profiler(self.model.loss, preds, labels, pred_lens, label_lens, self.training_info)

			status = "epoch: {}; iter: {}; loss_mean: {}; loss: {}".format(epoch_i, no, np.mean(self.training_info['Loss']), loss.data[0])

			print(status)

			self.profiler(loss.backward)
			self.profiler(self.model.opt.step)
			self.profiler(self.model.opt.zero_grad)

			self.periodic_check(no)

			del preds, pred_lens, loss, imgs, label_lens, labels, sample

			if (self.start_no + no)%self.config['test_now']==0 and (self.start_no + no)!=0:

				self.profiler(self.test_module, self.test_r, epoch_i, self.start_no + no)

			if (self.start_no + no) == len(self.train_data_loader) - 1:
				break
			
			self.profiler.step()

		self.profiler.dump()
	
		return True

		#Performs forward propogation and backward propogation, updates learning rate, can make changes in yaml file in real time
		#At regular intervals, model is tested(validation) and the corresponding testing accuracy is plotted
		#Similarly training accuracy also plotted
		#Saves the model regularly

	def test_r(self):

		self.profiler(log.info, 'Testing Mode')

		try:

			self.profiler(self.start_testing)

			with torch.no_grad():

				fps = {'images': 0, 'time_taken': 0}

				start = time.time()

				count = 0
				tp = 0
				avg_ed = 0

				for no in range(len(self.test_data_loader)):

					sample = self.profiler(self.test_data_loader.getitem, no)

					imgs = sample["img"]
					labels = sample["seq"].view(-1)
					label_lens = sample["seq_len"].int()

					fps['images'] += imgs.shape[0]

					if self.cuda:

						imgs = imgs.cuda()
					
					out = self.profiler(self.model, imgs, decode=True)
					gt = (sample["seq"].numpy() - 1).tolist()
					lens = sample["seq_len"].numpy().tolist()
					pos = 0
					for i in range(len(out)):

						gts = ''.join(self.test_data_loader.encoding_to_char[c] for c in gt[pos:pos+lens[i]])
						pos += lens[i]
						if gts == out[i]:
							tp += 1
						else:
							avg_ed += editdistance.eval(out[i], gts)
						count += 1

					fps['time_taken'] += time.time() - start

					del imgs, label_lens, labels, sample, out

					start = time.time()

				acc = tp / count
				avg_ed = avg_ed / count

				self.testing_info['Acc'] = acc

			self.profiler(log.info, 'Test Results\n\n')

			if self.mode =='train':

				if self.testing_info['Acc'] > self.model_best['Acc']:

					self.profiler(log.info, "New best model found")
					self.model_best['Acc'] = self.testing_info['Acc']
					
					self.profiler(self.model.save, no=0, epoch_i=0, info = self.testing_info, best=self.model_best, is_best=True)

				self.plot_testing['Acc'].append(self.testing_info['Acc'])

			self.profiler(log.info, '\nTesting Completed successfully: Average accuracy = ', self.testing_info['Acc'], 'Average FPS = ', fps['images']/fps['time_taken'])

			self.testing_info = {'Acc': 0, 'Count': 0, 'Keep_log': False}

			return True

		except KeyboardInterrupt:

			self.profiler(log.info, 'Testing Interrupted')

			return False

			#Tests model from testing data, saves best model if accuracy exceeds previous best
			#Saves model at regualar intervals
			#Records the time taken for model to output per image
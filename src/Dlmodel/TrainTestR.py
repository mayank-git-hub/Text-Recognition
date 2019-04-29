from .Dlmodel import Dlmodel
from ..helper.logger import Logger
from ..helper.utils import get_connected_components, strLabelConverter
import torch
import time
import random
import numpy as np
import editdistance
import tqdm as tqdm
import shutil
import os
import matplotlib.pyplot as plt

log = Logger()

class TrainTestR(Dlmodel):
	
	def __init__(self, config, mode = 'train', profiler=None):

		super().__init__(config, mode, profiler)
		self.seed()

		if config['varying_width']:
			self.list_or_tensor = 'list'
		else:
			self.list_or_tensor = 'tensor'

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

#Performs forward propogation and backward propogation, updates learning rate, can make changes in yaml file in real time
#At regular intervals, model is tested(validation) and the corresponding testing accuracy is plotted
#Similarly training accuracy also plotted
#Saves the model regularly

	def train_r(self):
		
		try:

			converter = strLabelConverter(self.train_data_loader.get_abc())
			spec_char_words = 0

			self.profiler(self.start_training)
			
			iterator = tqdm.tqdm(np.arange(self.start_no, self.config['steps']))
			
			for no in iterator:
				
				self.printed = 0
				self.logged = 0
				self.tested = 0

				sample = self.profiler(self.train_data_loader.getitem, no)

				list_or_tensor = self.list_or_tensor				
				
				if list_or_tensor == 'tensor':

					imgs = sample["img"]
					labels = sample["seq"].view(-1)
					label_lens = sample["seq_len"].int()
					true_labels = converter.decode(labels.data, label_lens.data, raw=False, target=True)

					if self.cuda:
						imgs = imgs.cuda()

					batch_size = imgs.shape[0]

					preds = self.model(imgs).cpu()
					pred_lens = torch.IntTensor([preds.size(0)] * batch_size).int()
					
				else:

					imgs = sample["img"]
					labels = sample["seq"]
					label_lens = sample["seq_len"]
					# print(labels[0], label_lens[0])
					# plt.imshow(imgs[0][0].data.cpu().numpy().transpose(1, 2, 0))
					# plt.show()
					# print(len(imgs), imgs[0].shape)
					# plt.imshow(imgs[])

					if self.cuda:

						imgs = [img.cuda() for img in imgs]	

						preds = [self.profiler(self.model, pred).cpu() for pred in imgs]
						pred_lens = [torch.IntTensor([len(pred)]) for pred in preds]

				save = True
				
				if save and len(os.listdir(self.config['dir']['Temp_save'])) < 60:

					lens = ([label_lens[i].numpy()[0] for i in range(len(label_lens))])
					gt = np.concatenate([labels[i].numpy() for i in range(len(labels))], axis=0)
					true_labels = converter.decode(torch.IntTensor(gt).data, torch.IntTensor(lens).data, raw=False,
					                               target=True)
					for l in true_labels:
						spec_char_words += 1

					tqdm.tqdm.write("Saving train images")

					for j in range(len(imgs)):

						text_act = true_labels[j].replace('/','forwardslash')
						# print(imgs[j])
						if self.channels == 3:
							plt.imsave(self.config['dir']['Temp_save']+'/'+text_act+'.png',imgs[j][0].data.cpu().numpy().transpose(1, 2, 0))
						else:
							plt.imsave(self.config['dir']['Temp_save']+'/'+text_act+'.png',imgs[j][0][0].data.cpu().numpy())

				loss = self.profiler(self.model.loss, preds, labels, pred_lens, label_lens, self.training_info)

				self.profiler(loss.backward)
				self.profiler(self.model.opt.step)
				self.profiler(self.model.opt.zero_grad)

				self.profiler(self.model.print_info, self.training_info, iterator)

				self.model.print_info(self.training_info, iterator)

				if no%self.config['update_config']==0 and no!=0:

					prev_config = self.config

					self.config = self.profiler(self.get_config, 'configs/text_config.yaml') #We can make changes to yaml file in real time
					self.train_data_loader._update_config(self.config)
					self.test_data_loader._update_config(self.config)
					self.model._update_config(self.config)

					if self.config['lr']!=prev_config['lr']:
						self.profiler(self.model.dynamic_lr, self.config['lr'], no)
						self.profiler(log.info, 'Learning Rate Changed from ', prev_config['lr'], ' to ', self.config['lr'])

				if no%self.config['model_save_iter'] == 0 and no!=0:
					tqdm.tqdm.write("Saving model")
					self.profiler(self.model.save, no=no, info = self.training_info, abc_len = len(self.train_data_loader.abc), best=self.model_best, is_best = False)

				del preds, pred_lens, loss, imgs, label_lens, labels, sample
				
				self.profiler.step()

				if no % self.config['test_now'] == 0 and no != 0:
					self.profiler(self.test_module, self.test_r)

		except KeyboardInterrupt:
			self.profiler.dump()
			return False
		self.profiler.dump()
		return True




#Tests model from testing data, saves best model if accuracy exceeds previous best
#Saves model at regualar intervals
#Records the time taken for model to output per image

	def test_r(self):

		self.profiler(self.start_testing)
		converter = strLabelConverter(self.test_data_loader.get_abc())

		with torch.no_grad():

			count = 0
			tp = 0
			avg_ed = 0

			# pbar = tqdm.tqdm(range(len(self.test_data_loader)//self.test_data_loader.batchsize))
			pbar = tqdm.tqdm(range(5))
			list_or_tensor = self.list_or_tensor

			to_print = {'true':[], 'predicted':[], 'images':[]}

			for no in pbar:

				sample = self.profiler(self.test_data_loader.getitem, no)

				if list_or_tensor == 'tensor':

					imgs = sample["img"]
					labels = sample["seq"].view(-1)
					label_lens = sample["seq_len"].int()

					true_labels = converter.decode(labels.data, label_lens.data, raw=False, target=True)

					if self.cuda:
						imgs = imgs.cuda()

					batch_size = imgs.shape[0]

					out = self.model(imgs)

					_, preds = out.max(2)
					preds = preds.transpose(1, 0).contiguous().view(-1)
					preds_size = torch.IntTensor([out.size(0)] * batch_size).int()
					predicted_final_label = converter.decode(preds.data, preds_size.data, raw=False)								

				else:
					imgs = sample["img"]
					labels = sample["seq"]
					label_lens = sample["seq_len"]

					gt = np.concatenate([labels[i].numpy() for i in range(len(labels))], axis=0)
					lens = ([label_lens[i].numpy()[0] for i in range(len(label_lens))])
					true_labels = converter.decode(torch.IntTensor(gt).data, torch.IntTensor(lens).data, raw=False, target=True)

					if self.cuda:

						imgs = [img.cuda() for img in imgs]	
				
					out = [self.profiler(self.model, img) for img in imgs]
					pred_argmax = [img.max(2)[1][:,0] for img in out]
					preds = torch.cat(pred_argmax)
					preds_lens = [torch.IntTensor([len(pred)]) for pred in out]
					preds_lens = torch.cat(preds_lens)

					predicted_final_label = converter.decode(preds.data, preds_lens.data, raw=False)

				for _p, _t in zip(predicted_final_label, true_labels):
					if _p == _t:
						tp += 1
					count += 1

				avg_ed += editdistance.eval(predicted_final_label, true_labels)

				to_print['true'] += true_labels
				to_print['predicted'] += predicted_final_label
				to_print['images'] += imgs

				del imgs, label_lens, labels, sample, out

			self.testing_info['Acc'] = tp/count
			self.testing_info['editdistance'] = avg_ed/count

			to_display = self.config['test_display']
			idx = np.random.choice(len(to_print['true']), min(to_display, len(to_print['true'])), replace=False)
			true = np.array(to_print['true'])[idx].tolist()
			pred = np.array(to_print['predicted'])[idx].tolist()
			# outss = np.array(to_print['images'])[idx].tolist()

			tqdm.tqdm.write('Actual: %s; Predicted: %s; Accuracy: %f; editdistance: %f' % (true, pred, tp/count, avg_ed/count))

			save = True
			
			if save:

				for j in range(to_display):

					# print(to_print['images'][idx[j]].shape)
					text_pred = pred[j].replace('/', 'forwardslash')
					text_act = true[j].replace('/', 'forwardslash')
					# ToDO - Ask Mithilesh Why?
					final_name = text_pred+'_'+text_act

					if self.channels == 3:
						plt.imsave(self.config['dir']['Temp_test_save']+'/'+final_name+'.png',to_print['images'][idx[j]][0].data.cpu().numpy().transpose(1,2,0))
					else:
						plt.imsave(self.config['dir']['Temp_test_save']+'/'+final_name+'.png',to_print['images'][idx[j]][0][0].data.cpu().numpy())

		if self.mode =='train':

			if self.testing_info['Acc'] > self.model_best['Acc']:

				self.model_best['Acc'] = self.testing_info['Acc']				
				self.profiler(self.model.save, no=0, info = self.testing_info, abc_len = len(self.train_data_loader.abc), best=self.model_best, is_best=True)
				tqdm.tqdm.write("Found new best model")

			self.plot_testing['Acc'].append(self.testing_info['Acc'])

		self.testing_info = {'Acc': 0, 'editdistance': 0}

		return True

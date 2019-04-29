import torch
import torch.nn as nn
import numpy as np
from ..helper.logger import Logger
import random
from tqdm import tqdm
import os
from ..helper.utils import get_connected_components, scores

log = Logger()

class model(nn.Module):

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def _update_config(self, config):

		self.config = config

	def update_lr(self, no):

		self.dynamic_lr(self.config['lr'], no)

		if no in self.config['lr']:
			self.set_lr(self.config['lr'][no])

	def set_lr(self, lr):

		if self.prev_lr == lr:
			pass

		for param_group in self.opt.param_groups:
			param_group['lr'] = lr

		self.prev_lr = lr

	def dynamic_lr(self, lr, no):

		stage = 1

		for k in lr.keys():
			if no>k and stage<k:
				stage = k

		self.set_lr(lr[stage])

	def print_info(self, info, iterator, status=''):
		# print(info['Loss'])
		if info['Keep_log']:
			status += "Learning Rate: {}| Training: Avg Accuracy: {}| Current accuracy: {}| Avg Loss: {}| Current Loss: {}".format(int(self.prev_lr*1000000)/1000000, int(np.mean(info['Acc'][-min(len(info['Acc']), 50):])*10000)/10000, int(info['Acc'][-1]*10000)/10000, int(np.mean(info['Loss'])*10000)/10000, int(info['Loss'][-1]*10000)/10000)
		else:

			status += "Avg Loss: {}| Avg Accuracy: {}".format(int(info['Loss']*10000)/10000, int(info['Acc']*10000)/10000)

		iterator.set_description(status)
		# print(status)
		#Saves the various results in logger file

	def save(self, no, info, abc_len, is_best=False, filename='checkpoint.pth.tar', best={}):
		
		if is_best:
			torch.save({
					'state_dict': self.state_dict(),
					'optimizer': self.opt.state_dict(),
					'seed': self.config['seed'],
					'best': best,},self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+filename)
			torch.save(info, self.config['dir']['Model_Output_Best']+'/'+str(no)+'_info_'+filename)
		else:
			x = os.listdir(self.config['dir']['Model_Output'])
			if len(x) == 2:
				for i in x:
					os.remove(self.config['dir']['Model_Output']+'/'+i)
			torch.save({
					'state_dict': self.state_dict(),
					'optimizer': self.opt.state_dict(),
					'seed': self.config['seed'],
					'best': best,},self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(abc_len)+'_'+filename)
			torch.save(info, self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(abc_len)+'_info_'+filename)


			#Saves the model at regular intervals when called in dl_model

	def load(self, path, path_info, Tr_te):

		checkpoint = torch.load(path)

		self.load_state_dict(checkpoint['state_dict'])

		if not self.config['optimizer_new']:
			self.opt.load_state_dict(checkpoint['optimizer'])

		if Tr_te == 'train':
		
			return torch.load(path_info)

		else:

			return torch.load(path_info)

		#Loads a pre-trained model

	def accuracy_(self, segmentation_predicted, data, real_target, contour_target, d_name, link_pred):

		"""
		link_pred_batch is np array with shape = [1, 16, height, width]
		Up-left, Up, Up-right, right, Down-right, Down, Down-left, left
		segmentation_predicted_batch is np array with shape = [1, 2, height, width]
		data_batch is np array with shape = [1, 3, height, width]
		real_target_batch is np array with shape = [1, 1, height, width]
		contour_target_batch is a list of size = [1, contours]
		"""

		precision_, recall_, f1_score_ = [], [], []

		if link_pred is not None:
			link_pred = link_pred[0].transpose(1, 2, 0)
		segmentation_predicted = segmentation_predicted[0].transpose(1, 2, 0)

		# plt.imshow(np.argmax(segmentation_predicted, axis=2))
		# plt.pause(0.1)
		data = data[0].transpose(1, 2, 0)
		real_target = real_target[0, 0]

		"""
		link_pred is np array with shape = [height, width, 16]
		Up-left, Up, Up-right, right, Down-right, Down, Down-left, left
		segmentation_predicted is np array with shape = [height, width, 2]
		data is np array with shape = [height, width, 3]
		real_target is np array with shape = [height, width]
		config is from read
		# print(len(contour_pred_dict['contour']))

		"""

		contour_pred_dict = get_connected_components(segmentation_predicted, data, real_target, self.config, d_name, False, link_pred=link_pred)

		# print(contour_pred_dict)

		precision, recall, f1_score = scores(contour_pred_dict, contour_target[0], threshold=0.5)
		precision_.append(precision)
		recall_.append(recall)
		f1_score_.append(f1_score)

		return np.mean(f1_score_)

	#Start tracking hard mining

	def hard_negative(self, pred, target, numpy_target_bg, numpy_target_fg):

		if self.config['hard_negative_mining']:
			loss_seg_temp = self.lossf(pred[:, 0:2], target)
		else:
			# loss_seg_temp = F.cross_entropy(pred[:, 0:2], target, weight=self.class_weight, reduction='none')
			loss_seg_temp = self.lossf(pred[:, 0:2], target)
		sorted_loss = np.argsort(-loss_seg_temp.squeeze().data.cpu().numpy()[numpy_target_bg])
		
		if self.config['hard_negative_mining']:
			if numpy_target_fg.shape[0]!= 0:
				num_bg = min(int(self.config['ratio']*numpy_target_fg.shape[0]), sorted_loss.shape[0])
				sorted_loss = sorted_loss[0:min(int(self.config['ratio']*numpy_target_fg.shape[0]), sorted_loss.shape[0])]
			else:
				num_bg = min(self.config['min_bg'], sorted_loss.shape[0])
				sorted_loss = sorted_loss[0:min(self.config['min_bg'], sorted_loss.shape[0])]
		else:
			num_bg = sorted_loss.shape[0]

		return sorted_loss, loss_seg_temp, num_bg

	#Start trackign link loss
	def link_loss(self, pred, numpy_target_fg, numpy_target_bg, sorted_loss, num_fg, num_bg, loss_seg, link=None, weight=None):

		if numpy_target_fg.shape[0]!= 0:

			# print(num_bg)

			if self.config['weight_bbox']:
				loss_positive = (weight*loss_seg[numpy_target_fg]).sum()/(num_fg + num_bg)
			else:
				loss_positive = (loss_seg[numpy_target_fg]).sum()/(num_fg + num_bg)

			loss_negative = loss_seg[numpy_target_bg][sorted_loss].sum()/(num_fg + num_bg)
			loss_seg = self.config['sed_link']*(self.config['pos_neg']*loss_positive + loss_negative)
		
		else:

			loss_negative = loss_seg[numpy_target_bg][sorted_loss].sum()/(num_fg + num_bg)
			loss_seg = self.config['sed_link']*loss_negative

		if numpy_target_fg.shape[0]!= 0:

			if self.config['link']:
	
				positive = 0
				negative = 0

				positive_reduce_sum = 0
				negative_reduce_sum = 0
				
				for i in range(8):
					
					if self.config['weight_bbox']:
						temp_loss = weight*self.lossf(pred[numpy_target_fg, 2*i+2:2*i+4], link[:, i])
					else:
						temp_loss = self.lossf(pred[numpy_target_fg, 2*i+2:2*i+4], link[:, i])
						
					link_ = link.data.cpu().numpy()
					positive_link = np.where(link_[:,i] == 1)[0]
					negative_link = np.where(link_[:, i] == 0)[0]

					positive += temp_loss[positive_link].sum()
					negative += temp_loss[negative_link].sum()
					
					positive_reduce_sum += positive_link.shape[0]
					negative_reduce_sum += negative_link.shape[0]

				loss_link = 0

				if negative_reduce_sum!=0:
					loss_link += negative/negative_reduce_sum
				if positive_reduce_sum!=0:
					loss_link += positive/positive_reduce_sum 

				# print(positive.item(), negative.item(), positive_reduce_sum, negative_reduce_sum, loss_link.item(), loss_seg.item())

				return loss_link + loss_seg

			else:

				return loss_seg
		else:

			return loss_seg

	def loss(self, data_batch_big, pred_big, target_big, contour_big, info, d_name, link_big=None, weight_big=None):

		loss_c = 0

		numpy_target_fg_big = []

		if self.config['weight_bbox'] and self.config['link']:
			iterator = zip(data_batch_big, pred_big, target_big, link_big, weight_big)
		elif self.config['weight_bbox']:
			iterator = zip(data_batch_big, pred_big, target_big, weight_big)
		elif self.config['link']:
			iterator = zip(data_batch_big, pred_big, target_big, link_big)
		else:
			iterator = zip(data_batch_big, pred_big, target_big)

		iterator = list(iterator)
		for no_ in range(len(iterator)):
			
			if self.config['weight_bbox'] and self.config['link']:
				data_batch, pred, target, link, weight = iterator[no_]
			elif self.config['weight_bbox']:
				data_batch, pred, target, weight = iterator[no_]
			elif self.config['link']:
				data_batch, pred, target, link = iterator[no_]
			else:
				data_batch, pred, target = iterator[no_]

			b, ch, h, w = pred.size()

			target = target.transpose(1, 3).contiguous().view(b*h*w).long()
			numpy_target = target.data.cpu().numpy()
			numpy_target_fg = np.where(numpy_target == 1)[0] #Returns numpy array of pixels where text is present
			numpy_target_fg_big.append(numpy_target_fg.shape[0])
			num_fg, num_bg = numpy_target_fg.shape[0], 0

			if numpy_target_fg.shape[0]!= 0:

				if self.config['link']:
					link_pred_batch = pred[:, 2:, :, :].data.cpu().numpy()
					link = link.transpose(1, 3).contiguous().view(b*h*w, 8).long()[numpy_target_fg, :]#.view() is like reshape
				if self.config['weight_bbox']:
					weight = weight.transpose(1, 3).contiguous().view(b*h*w)[numpy_target_fg]
				#Only consider weights where the pixels in target are present in text
				#Similar for links

			pred = pred.transpose(1, 3).contiguous().view(b*w*h, ch)
			numpy_target_bg = np.where(numpy_target == 0)[0]
			#Hard negative mining done. Log here
			sorted_loss, loss_seg, num_bg = self.profiler(self.hard_negative, pred, target, numpy_target_bg, numpy_target_fg, attr='hard_negative')  
			#Calculated link loss. Log here
			if self.config['weight_bbox'] and self.config['link']:
				loss_c += self.profiler(self.link_loss, pred, numpy_target_fg, numpy_target_bg, sorted_loss, num_fg, num_bg, loss_seg, link=link, weight=weight, attr='link_loss')
			elif self.config['weight_bbox']:
				loss_c += self.profiler(self.link_loss, pred, numpy_target_fg, numpy_target_bg, sorted_loss, num_fg, num_bg, loss_seg, link=None, weight=weight, attr='link_loss')
			elif self.config['link']:
				loss_c += self.profiler(self.link_loss, pred, numpy_target_fg, numpy_target_bg, sorted_loss, num_fg, num_bg, loss_seg, link=link, weight=None, attr='link_loss')
			else:
				loss_c += self.profiler(self.link_loss, pred, numpy_target_fg, numpy_target_bg, sorted_loss, num_fg, num_bg, loss_seg, link=None, weight=None, attr='link_loss')

		loss_c = loss_c/len(data_batch_big)

		if info['Keep_log']:

			# def accuracy_(self, link_pred_batch, segmentation_predicted_batch, data_batch, real_target_batch, contour_target_batch):

			"""
				link_pred_batch is torch.FloatTensor with shape = [batch_size, 16, height, width]
				Up-left, Up, Up-right, right, Down-right, Down, Down-left, left
				segmentation_predicted_batch is torch.FloatTensor with shape = [batch_size, 2, height, width]
				data_batch is torch.FloatTensor with shape = [batch_size, 3, height, width]
				real_target_batch is torch.FloatTensor with shape = [batch_size, 1, height, width]
				contour_target_batch is a list of size = [batch_size, contours]
			"""
			if info['Count']%self.config['accuracy_log'] == 0:

				acc = 0
				for i in range(len(data_batch_big)):

					if numpy_target_fg_big[i]!=0:
						if self.config['link']:
							acc += self.profiler(self.accuracy_, pred_big[i][:, 0:2, :, :].data.cpu().numpy(), data_batch_big[i], target_big[i].data.cpu().numpy(), [contour_big[i]], d_name[i], pred_big[i][:, 2:, :, :].data.cpu().numpy())
						else:
							acc += self.profiler(self.accuracy_, pred_big[i][:, 0:2, :, :].data.cpu().numpy(), data_batch_big[i], target_big[i].data.cpu().numpy(), [contour_big[i]], d_name[i], None)
					else:
						if np.any(np.argmax(pred_big[i][:, 0:2, :, :].data.cpu().numpy(), axis=1) == 1):
							acc += 0
						else:
							acc += 1
				info['Acc'].append(acc/len(data_batch_big))
			info['Loss'].append(loss_c.data.cpu().numpy())
				# info['Link_Loss'].append(loss_link.data.cpu().numpy())
				# info['Seg_loss'].append(loss_seg.data.cpu().numpy())

		else:
			acc = 0
			for i in range(len(data_batch_big)):
				if numpy_target_fg_big[i]!=0:
					if self.config['link']:
						acc += self.profiler(self.accuracy_, pred_big[i][:, 0:2, :, :].data.cpu().numpy(), data_batch_big[i], target_big[i].data.cpu().numpy(), [contour_big[i]], d_name[i], pred_big[i][:, 2:, :, :].data.cpu().numpy())
					else:
						acc += self.profiler(self.accuracy_, pred_big[i][:, 0:2, :, :].data.cpu().numpy(), data_batch_big[i], target_big[i].data.cpu().numpy(), [contour_big[i]], d_name[i], None)
				else:
					if np.any(np.argmax(pred_big[i][:, 0:2, :, :].data.cpu().numpy(), axis=1) == 1):
						acc += 0
					else:
						acc += 1

			acc = acc/len(data_batch_big)
			info['Acc'] = (acc + info['Count']*info['Acc'])/(info['Count']+1)
			info['Loss'] = (loss_c.data.cpu().numpy() + info['Count']*info['Loss'])/(info['Count']+1)
			# info['Link_Loss'] = (loss_link.data.cpu().numpy() + info['Count']*info['Loss'])/(info['Count']+1)
			# info['Seg_loss'] = (loss_seg.data.cpu().numpy() + info['Count']*info['Loss'])/(info['Count']+1)

		info['Count'] += 1

		return loss_c

from __future__ import division 
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import time
import os
import pickle
from torch.autograd import Variable
def line(p1, p2):

	A = (p1[1] - p2[1])
	B = (p2[0] - p1[0])
	C = (p1[0]*p2[1] - p2[0]*p1[1])
	return A, B, -C

def line_intersection(l1, l2):

	# l1 is a numpy array with shape 2, 2 with each row containing x, y

	d  = l1[0] * l2[1] - l1[1] * l2[0]
	dx = l1[2] * l2[1] - l1[1] * l2[2]
	dy = l1[0] * l2[2] - l1[2] * l2[0]
	if d != 0:
		x = (dx / d).astype(np.int64)
		y = (dy / d).astype(np.int64)
		return [x,y]
	else:
		return None

def inside_point(point, rect):

	# point is a list (x, y)
	# rect is a contour with shape [4, 2]

	rect = rect.reshape([4, 1, 2]).astype(np.int64)

	dist = cv2.pointPolygonTest(rect,(point[0], point[1]),True)

	if dist>0:
		# print(dist)
		return True
	else:
		return False

def intersection_union(cont1, cont2):

	# Assuming both contours are rectangle with shape [4, 1, 2]

	cont1 = cont1.reshape([cont1.shape[0], 2]).astype(np.float64)
	cont2 = cont2.reshape([cont2.shape[0], 2]).astype(np.float64)

	intersection_points = []

	line_i = [line(cont1[i], cont1[(i+1)%4]) for i in range(4)]
	line_j = [line(cont2[j], cont2[(j+1)%4]) for j in range(4)]

	min_i_x = [np.min(cont1[[i, (i+1)%4], 0]) for i in range(4)]
	max_i_x = [np.max(cont1[[i, (i+1)%4], 0]) for i in range(4)]
	min_j_x = [np.min(cont2[[j, (j+1)%4], 0]) for j in range(4)]
	max_j_x = [np.max(cont2[[j, (j+1)%4], 0]) for j in range(4)]

	min_i_y = [np.min(cont1[[i, (i+1)%4], 1]) for i in range(4)]
	max_i_y = [np.max(cont1[[i, (i+1)%4], 1]) for i in range(4)]
	min_j_y = [np.min(cont2[[j, (j+1)%4], 1]) for j in range(4)]
	max_j_y = [np.max(cont2[[j, (j+1)%4], 1]) for j in range(4)]

	for i in range(4):
		if inside_point(cont1[i], cont2):
			intersection_points += [cont1[i]]
		if inside_point(cont2[i], cont1):
			intersection_points += [cont2[i]]

	for i in range(4):
		for j in range(4):
			point = line_intersection(line_i[i], line_j[j])
			cond1 = point is not None
			if not cond1:
				continue
			cond2 = point[0] >= min_i_x[i] and point[0] >= min_j_x[j] and point[0] <= max_i_x[i] and point[0] <= max_j_x[j]
			if not cond2:
				continue
			cond3 = point[1] >= min_i_y[i] and point[1] >= min_j_y[j] and point[1] <= max_i_y[i] and point[1] <= max_j_y[j]
			if not cond3:
				continue
			intersection_points += [point]
	
	if len(intersection_points) < 3:
		return 1, 0

	contour = np.array(intersection_points).reshape([len(intersection_points), 1, 2]).astype(np.int64)
	contour = cv2.convexHull(contour)
	intersection_val = cv2.contourArea(contour)
	if intersection_val!=0:
		union_val = (cv2.contourArea(cont1.reshape([4, 1, 2]).astype(np.int64))+cv2.contourArea(cont2.reshape([4, 1, 2]).astype(np.int64)) - intersection_val)
		return union_val, intersection_val
	else:
		return 1, 0

def precision_recall_fscore(true_positive, false_positive, false_negative):

	if true_positive + false_negative == 0:
		recall = 0
	else:
		recall = true_positive/(true_positive + false_negative)

	if true_positive + false_positive == 0:
		precision = 0
	else:
		precision = true_positive/(true_positive + false_positive)
	
	if recall == 0 or precision==0:
		f1_score = 0
	else:
		f1_score = 2*(recall * precision) / (recall + precision)

	return precision, recall, f1_score

def scores(contour_pred, contour_target, threshold, text=False, text_pred=False, text_target=False, multiple=False):

	already_checked = np.zeros(len(contour_target)).astype(np.bool)

	false_positive = 0
	true_positive = 0

	for i in range(len(contour_pred)):
		
		found = False

		for j in range(len(contour_target)):

			if already_checked[j] == True:
				continue
			
			union, inter = intersection_union(contour_pred[i], contour_target[j])
			if inter/union > threshold:
				found = True
				already_checked[j] = True
				break
		if found:
			if text:
				if text_target[j] is None:
					continue
				if text_target[j].lower() == text_pred[i].lower():
					true_positive += 1
				else:
					false_positive += 1
			else:
				true_positive += 1
		else:
			false_positive += 1

	false_negative = already_checked.shape[0] - np.sum(already_checked.astype(np.float64))

	if multiple:

		return true_positive, false_positive, false_negative

	else:

		return precision_recall_fscore(true_positive, false_positive, false_negative)

	#This calculates precision=TP/(TP+FP)
	#recall=TP/(TP+FN)
	#Fscore is harmonic mean of precision and recall

def overlap_remove(contour,threshold):

	contour = np.array(contour)
	to_remove = np.zeros([contour.shape[0]])

	for i in range(contour.shape[0]):

		for j in range(i+1,contour.shape[0]):

			if to_remove[j] == 1:
				continue

			union, inter = intersection_union(contour[i], contour[j])
			cnt_a_1, cnt_a_2 = cv2.contourArea(contour[i]), cv2.contourArea(contour[j])	
			
			if (inter/cnt_a_1) > threshold:
				if (inter/cnt_a_2) > threshold:
					if cnt_a_2 > cnt_a_1:
						to_remove[i] = 1
					else:
						to_remove[j] = 1
				else:
					to_remove[i] = 1
			elif (inter/cnt_a_2) > threshold:
				to_remove[j] = 1

	return contour[np.where(to_remove == 0)[0]]

def get_rotated_bbox(contours):

	return_contours = []

	for i in range(len(contours)):

		cnt = np.array(contours[i]).astype(np.int64)
		rect = cv2.minAreaRect(cnt)
		return_contours.append(cv2.boxPoints(rect).astype(np.int64).reshape([4, 1, 2]))

	return return_contours

	#Creates bounding box using minAreaRect, gets the points, the rectangle need not be parallel to x-axis

def remove_small_boxes(contours, min_area, max_area=None):

	"""
	input - contour, min_area, max_are
	return - thresholded contour
	"""

	contours = get_rotated_bbox(contours)

	return_contours = []

	for i in range(len(contours)):
		area = cv2.contourArea(contours[i])
		if area > min_area:
			if max_area!=None:
				if area < max_area:
					return_contours.append(contours[i])
			else:
				return_contours.append(contours[i])

	return return_contours

	#Removes contours whose area is smaller than specified value or larger than max_area(if specified), and returns the remaining contours

def get_connected_components(segmentation_predicted, data, real_target, config, d_name, show=True, path=None, link_pred=None, resize=None):

	"""
	link_pred is np array with shape = [height, width, 16]
	Up-left, Up, Up-right, right, Down-right, Down, Down-left, left
	segmentation_predicted is np array with shape = [height, width, 2]
	data is np array with shape = [height, width, 3]
	real_target is np array with shape = [height, width]
	config is from read_config
	"""
	"""
	Function takes in input of pixels (by segmentation) and links accurately classified 
	and returns the countour of various connected components.
	"""

	# TODO - vary threshold for model output 
	# Discretizing of scores

	# Softmax
	# print(d_name)

	#Softamx applied on predicted segmentation

	segmentation_predicted = np.exp(segmentation_predicted)
	segmentation_predicted = segmentation_predicted/np.sum(segmentation_predicted, axis=2)[:, :, None]
	pixel_pred = (segmentation_predicted[:, :, 1] > config['metadata'][d_name]['segmentation_thresh']).astype(np.float32)

	if config['link']:
		link_pred = np.exp(link_pred)
		link_predicted_8_channel = np.zeros([link_pred.shape[0], link_pred.shape[1], 8])	
		for i in range(8):
			
			link_pred[:, :, 2*i:2*i+2] = link_pred[:, :, 2*i:2*i+2]/np.sum(link_pred[:, :, 2*i:2*i+2], axis=2)[:, :, None]
			link_predicted_8_channel[:, :, i] = (link_pred[:, :, 2*i+1] > config['metadata'][d_name]['link_thresh']).astype(np.float32)

			#For each in link_predicted_8_channel, there are 2 channels of link_pred, normalised mean given to all three
			#If this is above threshold, then make the channel value 1, else 0

	# Initialization of some useful values
	image_size = pixel_pred.shape
	target = np.zeros([image_size[0], image_size[1], 3])
	target[:, :, 0] = pixel_pred*255
	target = target.astype(np.uint8)

	if config['link']:
	
		# Initialization of Graph
		moves = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
		edges = np.zeros(link_predicted_8_channel.shape)
		row, column = np.where(pixel_pred==1)
		g = nx.Graph()
		g.add_nodes_from(zip(row,column))
		link_predicted_8_channel = link_predicted_8_channel*pixel_pred[:, :, None]
		link_boundary = (np.any(link_predicted_8_channel==0, axis=2).astype(np.float32)*pixel_pred*255).astype(np.uint8)

		# Processing to allow us to use nx.connected_components
		pixel_pred = np.pad(pixel_pred, 1, 'constant', constant_values = 0)
		for i in range(8):
			x,y = moves[i]
			edges[:,:,i] = link_predicted_8_channel[:,:,i]*pixel_pred[1+x:1+x+image_size[0], 1+y:1+y+image_size[1]]

		for i in range(8):
			row, column = np.where(edges[:,:,i]==1)
			g_edges1 = list(zip(row,column))
			g_edges2 = list(zip(row+moves[i][0],column+moves[i][1]))
			g.add_edges_from(zip(g_edges1,g_edges2))

		# Set of connected components

		connected = list(nx.connected_components(g))

		# Converting to fit the countour half of the program

		for i in range(len(connected)):
			connected[i] = np.flip(np.array(list(connected[i])).reshape([len(connected[i]), 1, 2]), axis=-1)

	else:

		connected, _ = cv2.findContours(pixel_pred.copy().astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	if show:

		image = np.zeros([image_size[0], image_size[1], 3]).astype(np.uint8)
		if len(connected) < 30000:

			if resize is not None:
				connected = resize['function'](resize['base_r'], resize['base_c'], list(connected), resize['original_image_shape'])

			connected = remove_small_boxes(connected, 100)
			connected = overlap_remove(connected,0.2)

			for i in range(len(connected)):

				color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
				cv2.drawContours(image, [connected[i]], -1, color, cv2.FILLED)

			if str(data.dtype) == 'uint8':
				images = data
			else:
				images = (data*255).astype(np.uint8)
		
			if real_target is not None:

				real_target = (real_target*255).astype(np.uint8)

				contours, hierarchy = cv2.findContours(real_target.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

				images = images.copy()
				cv2.drawContours(images, contours, -1, (0, 255, 0), 1)
				if len(connected)!=0:
					cv2.drawContours(images, connected, -1, (255, 0, 0), 1)

			else:

				images = images.copy()
				cv2.drawContours(images, connected, -1, (0, 255, 0), 1)

			plt.clf()
			
			if path == None:

				if not os.path.exists(config['dir']['Exp']+'/Output_Train'):
					os.mkdir(config['dir']['Exp']+'/Output_Train')

				plt.imsave(config['dir']['Exp']+'/Output_Train/contour_on_image_train.png', images)
				plt.imsave(config['dir']['Exp']+'/Output_Train/contour_blank_train.png', image)

				if config['link']:
					plt.imsave(config['dir']['Exp']+'/Output_Train/boundary_train.png', link_boundary)
					plt.imsave(config['dir']['Exp']+'/Output_Train/link_train_UL.png', (link_predicted_8_channel[:, :, 0]*255).astype(np.uint8))
					plt.imsave(config['dir']['Exp']+'/Output_Train/link_train_U.png', (link_predicted_8_channel[:, :, 1]*255).astype(np.uint8))
					plt.imsave(config['dir']['Exp']+'/Output_Train/link_train_UR.png', (link_predicted_8_channel[:, :, 2]*255).astype(np.uint8))
					plt.imsave(config['dir']['Exp']+'/Output_Train/link_train_R.png', (link_predicted_8_channel[:, :, 3]*255).astype(np.uint8))
					plt.imsave(config['dir']['Exp']+'/Output_Train/link_train_BR.png', (link_predicted_8_channel[:, :, 4]*255).astype(np.uint8))
					plt.imsave(config['dir']['Exp']+'/Output_Train/link_train_B.png', (link_predicted_8_channel[:, :, 5]*255).astype(np.uint8))
					plt.imsave(config['dir']['Exp']+'/Output_Train/link_train_BL.png', (link_predicted_8_channel[:, :, 6]*255).astype(np.uint8))
					plt.imsave(config['dir']['Exp']+'/Output_Train/link_train_L.png', (link_predicted_8_channel[:, :, 7]*255).astype(np.uint8))
					plt.imsave(config['dir']['Exp']+'/Output_Train/link_train_L.png', (link_predicted_8_channel[:, :, 7]*255).astype(np.uint8))
				plt.imsave(config['dir']['Exp']+'/Output_Train/segmentation_argmax_train.png', target)
				plt.imsave(config['dir']['Exp']+'/Output_Train/segmentation_continuous_train.png', (segmentation_predicted[:, :, 1]*255).astype(np.uint8))

			else:
				# plt.imsave(path+'.png', images)
				if not os.path.exists(path):
					os.mkdir(path)


				plt.imsave(path+'/contour_on_image.png', images)
				plt.imsave(path+'/contour_blank.png', image)
				if config['link']:
					plt.imsave(path+'/boundary.png', link_boundary)
					plt.imsave(path+'/link_UL.png', (link_predicted_8_channel[:, :, 0]*255).astype(np.uint8))
					plt.imsave(path+'/link_U.png', (link_predicted_8_channel[:, :, 1]*255).astype(np.uint8))
					plt.imsave(path+'/link_UR.png', (link_predicted_8_channel[:, :, 2]*255).astype(np.uint8))
					plt.imsave(path+'/link_R.png', (link_predicted_8_channel[:, :, 3]*255).astype(np.uint8))
					plt.imsave(path+'/link_BR.png', (link_predicted_8_channel[:, :, 4]*255).astype(np.uint8))
					plt.imsave(path+'/link_B.png', (link_predicted_8_channel[:, :, 5]*255).astype(np.uint8))
					plt.imsave(path+'/link_BL.png', (link_predicted_8_channel[:, :, 6]*255).astype(np.uint8))
					plt.imsave(path+'/link_L.png', (link_predicted_8_channel[:, :, 7]*255).astype(np.uint8))
					plt.imsave(path+'/link_L.png', (link_predicted_8_channel[:, :, 7]*255).astype(np.uint8))
				plt.imsave(path+'/segmentation_argmax.png', target)
				plt.imsave(path+'/segmentation_continuous.png', (segmentation_predicted[:, :, 1]*255).astype(np.uint8))

			return connected
		else:

			print(len(connected), ' bounding boxes are too high to be normal')
			return np.array([])

	else:
		connected = remove_small_boxes(connected, 100)
		connected = overlap_remove(connected, 0.2)
		return connected

#arg1 = Path to predicted folder, arg2 = Path to target folder, arg3 = threshold for IOU to consider positive

def get_f_score(path_pred, path_tar, threshold, text):

	predicted_filename = []
	target_filename = []
	for file_ in os.listdir(path_pred):
		if file_.endswith(".pkl"):
			predicted_filename.append(file_)
	for file_ in os.listdir(path_tar):
		if file_.endswith(".pkl"):
			target_filename.append(file_)	

	true_positive, false_positive, false_negative = 0, 0, 0

	for j,f2 in enumerate(target_filename):
		
		if os.path.exists(path_pred+'/'+f2):
			bbox_predict, text_pred = pickle.load(open(path_pred + '/' + f2,'rb'))
			bbox_target, text_tar = pickle.load(open(path_tar + '/' + f2,'rb'))
			temp_true_positive, temp_false_positive, temp_false_negative = scores(bbox_predict,bbox_target,threshold, text=text, text_pred=text_pred, text_target=text_tar, multiple=True)
			true_positive += temp_true_positive
			false_positive += temp_false_positive
			false_negative += temp_false_negative
		else:
			print ('File ' + f2.lower() + ' not found')

	print(precision_recall_fscore(true_positive, false_positive, false_negative)[2])

	return precision_recall_fscore(true_positive, false_positive, false_negative)[2]


def one_hot(index, classes):
	size = index.size() + (classes,)
	view = index.size() + (1,)

	mask = torch.Tensor(*size).fill_(0).to(index.get_device())
	index = index.view(*view)
	ones = 1.

	if isinstance(index, Variable):
		ones = torch.Tensor(index.size()).fill_(1).to(index.get_device())
		mask = mask.to(index.get_device())

	return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

	def __init__(self, gamma=1, eps=1e-7):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.eps = eps

	def forward(self, input, target):
		y = one_hot(target, input.size(-1))
		logit = F.softmax(input, dim=-1)
		logit = logit.clamp(self.eps, 1. - self.eps)

		loss = -1 * y * torch.log(logit) # cross entropy
		loss = loss * (1 - logit) ** self.gamma # focal loss

		return loss.mean(dim=1)

def homographic_rotation(angles, image, contours, path, index):
	"""
	this function output rotated image along the angles specified (x,y,z) and increases the shape of image to size by 
	having zeros at the other places.
	contour - np.array of contours which are np.arrays
	
	add a script for translations too, can also apply affine transfromations - though will affine transformations be
	helpful?

	interpolation to increase size of image? scaling?
	"""
	for i in range(len(contours)):

		if len(contours[i].shape) == 4:
			contours[i] = contours[i].squeeze(2)

	size = np.array([np.sqrt(image.shape[0]**2 + image.shape[1]**2), np.sqrt(image.shape[0]**2 + image.shape[1]**2)]).astype(np.int32)
	angles = np.deg2rad(angles)
	x = angles[0]
	y = angles[1]
	z = angles[2]
	h,w,n = image.shape

	# print(np.array(contours[0]).shape)
	# check contours shape and make compatable

	rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
	ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
	rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
	r = np.dot(rx, ry.dot(rz).astype(np.float32))
	t1 = np.array([[1, 0, size[1]/2-w/2], [0, 1, size[0]/2-h/2]], np.float32)
	t2 = np.array([[1, 0, size[1]/2], [0, 1, size[0]/2]], np.float32)

	image = cv2.warpAffine(image ,t1 ,(size[1], size[0]))

	coordinates_image = np.array([[size[1]/2-w/2,size[1]/2-w/2,size[1]/2+w/2,size[1]/2+w/2],[size[0]/2-h/2,size[0]/2+h/2,size[0]/2-h/2,size[0]/2+h/2]], np.float32).T
	coordinates_output = np.array([[-w/2, -w/2, w/2, w/2], [-h/2, h/2, -h/2, h/2], [0,0,0,0]], np.float32)

	new_contours = [[] for i in range(len(contours))]

	coordinates_output = r.dot(coordinates_output)
	coordinates_output[2] = [1,1,1,1]
	coordinates_output = (t2.dot(coordinates_output)).T
	coordinates_output = coordinates_output.astype(np.float32)

	
	m = cv2.getPerspectiveTransform(coordinates_image, coordinates_output)
	output_image = cv2.warpPerspective(image, m, (size[1], size[0])).astype(np.uint8)

	coordinates_output = coordinates_output.astype(np.int32)

	output_image = output_image[coordinates_output[:,1].min():coordinates_output[:,1].max(), coordinates_output[:,0].min():coordinates_output[:,0].max(), :]

	new_origin_x =  coordinates_output[:,1].min() # This is the vertical axis in image
	new_origin_y =  coordinates_output[:,0].min() # This is the horizontal axis

	all_cords_r, all_cords_c = np.where(np.sum(output_image, axis=2)!=0)

	average_color = np.mean(image).astype(np.uint8)
	output_image[output_image==0] = average_color

	for j in range(len(contours)):
		for i in range(len(contours[j])):
			if contours[j][i].shape[0]!=0:
				new_contour = np.array([[0,0,0,0], [0,0,0,0], [1,1,1,1]])
				if len(contours[j][i].shape)!=2:
					print(contours[j][i].shape)
					print(path, index)
					print(contours[j][i])
					plt.imshow(image)
					plt.show()
					print(contours[j][i].shape)
				new_contour[:2] = contours[j][i].T
				new_contour = t1.dot(new_contour).T
				new_contour = new_contour.astype(np.float32)
				new_contour = cv2.perspectiveTransform(new_contour[None,:,:], m)
				new_contour[:,:, 1] -= new_origin_x
				new_contour[:,:, 0] -= new_origin_y
				
				new_contour = np.array(new_contour).astype(np.int32)

				new_contours[j].append(new_contour)
			else:
				new_contours[j].append(np.array([]))

	for i in range(len(contours)):

		new_contours[i] = np.array(new_contours[i]).astype(np.int32)
		new_contours[i] = new_contours[i].reshape([new_contours[i].shape[0], 4, 1, 2])

		if new_contours[i].shape[0]!=0:
			new_contours[i][:, :, :, 0] -=all_cords_c.min()
			new_contours[i][:, :, :, 1] -=all_cords_r.min()

	return output_image[all_cords_r.min(): all_cords_r.max(), all_cords_c.min(): all_cords_c.max()], new_contours

class strLabelConverter(object):
	"""Convert between str and label.

	NOTE:
		Insert `blank` to the alphabet for CTC.

	Args:
		alphabet (str): set of the possible characters.
		ignore_case (bool, default=True): whether or not to ignore all of the case.
	"""

	def __init__(self, alphabet, ignore_case=True):
		self._ignore_case = ignore_case
		if self._ignore_case:
			alphabet = alphabet.lower()
		self.alphabet = alphabet + '-'  # for `-1` index

		self.dict = {}
		for i, char in enumerate(alphabet):
			# NOTE: 0 is reserved for 'blank' required by wrap_ctc
			self.dict[char] = i + 1

	def encode(self, text):
		"""Support batch or single str.

		Args:
			text (str or list of str): texts to convert.

		Returns:
			torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
			torch.IntTensor [n]: length of each text.
		"""
		if isinstance(text, str):
			text = [
				self.dict[char.lower() if self._ignore_case else char]
				for char in text
			]
			length = [len(text)]
		elif isinstance(text, collections.Iterable):
			length = [len(s) for s in text]
			text = ''.join(text)
			text, _ = self.encode(text)
		return (torch.IntTensor(text), torch.IntTensor(length))

	def decode(self, t, length, raw=False, target=False):
		"""Decode encoded texts back into strs.

		Args:
			torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
			torch.IntTensor [n]: length of each text.

		Raises:
			AssertionError: when the texts and its length does not match.

		Returns:

			text (str or list of str): texts to convert.
		"""
		# print(target)
		if length.numel() == 1:
			length = length[0]
			assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
			if raw:
				return ''.join([self.alphabet[i - 1] for i in t])
			else:
				char_list = []
				for i in range(length):
					if not target:
						if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
							char_list.append(self.alphabet[t[i] - 1])
					else:
						if t[i] != 0 :

							try:
								char_list.append(self.alphabet[t[i] - 1])
							except:
								print('Maybe delete cache. You may have changed the length of self.abc(alphabets)')

				return ''.join(char_list)
		else:
			# batch mode
			assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
			texts = []
			index = 0
			for i in range(length.numel()):
				l = length[i]
				texts.append(
					self.decode(
						t[index:index + l], torch.IntTensor([l]), raw=raw, target=target))
				index += l
			return texts
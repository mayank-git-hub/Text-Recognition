from .read_yaml import read_yaml
import json
import time
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
import torch
import random

class Profiler():

	def __init__(self, config):

		self.config = config
		self.seed()
		self.plots_dir = self.config['dir']['Plots']
		self.plot_counter = 0
		self.write_path = self.config['dir']['Exp']+'/profiler.json'
		reset = self.config['Profiler']['reset']
		self.step_no = 0

		if reset:
			self.reset_profiler()

		if os.path.exists(self.write_path):
			with open(self.write_path, 'r') as f:
				data_in = json.load(f)
				self.step_no, self.attrs, self.total_time = data_in['steps'], data_in['profile'], data_in['total_time']
		else:
			self.attrs = []
			self.step_no = 0
			self.total_time = time.time()

		self.stack = []
		self.found = False

		self.dump_interval = 1000

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def dump(self):

		print("Dumping json")
		f = open(self.write_path, 'w')
		json.dump({'steps': self.step_no, 'profile': self.attrs, 'total_time': time.time() - self.total_time}, f, indent=4)
		f.close()
		self.plot()

		# Initialising with attrs = self.attrs, base_folder = self.config['dir']['Profiler'], base_time = self.attr

	def plot(self):

		current_json = {'steps': self.step_no, 'profile': self.attrs, 'total_time': time.time() - self.total_time}
		for i in self.attrs:
			self.plot_recursive(i.copy(), self.config['dir']['Profiler'], i['time'], current_json['steps'])#/current_json['steps']

	def plot_recursive(self, attrs, base_folder, base_time, div_number):

		# Creating a folder named aatr+'_children' which contains the plot of the attr children, a plot showing time division at that level

		labels = []
		values = []
		sum_time = 0
		
		for attr in attrs['children']:
			labels.append(attr['name'])
			values.append(attr['time'])
			sum_time += attr['time']

		arg = np.argsort(values)

		labels = np.array(labels)[arg].tolist()
		values = np.array(values)[arg]

		plt.pie(values, labels=labels, autopct='%1.1f%%')#, radius=len(labels)
		plt.xlabel(attrs['name']+'\n'+'Unaccounted Time: '+str((base_time - sum_time))+'\nTotal Time: '+str(base_time))#/div_number

		if not os.path.exists(base_folder+'/'+str(attrs['name'])):
			os.mkdir(base_folder+'/'+str(attrs['name']))

		plt.savefig(base_folder+'/'+str(attrs['name'])+'/'+str(attrs['name'])+'.png')

		plt.clf()

		if len(attrs['children']) != 0:
			for i in attrs['children']:
				if len(i['children']):
					self.plot_recursive(i.copy(), base_folder+'/'+str(attrs['name']), i['time'], div_number)#/div_number

	def update_value(self, attr, value, d):

		for element in d:
			if element['name'] == attr:
				element['time'] += value
				element['count'] += 1
				return

			elif len(element['children']):
				self.update_value(attr, value, element['children'])

	def find_exists(self, attr, d):

		for element in d:
			if element['name'] == attr:
				self.found = True
				return
			elif len(element['children']):
				self.find_exists(attr, element['children'])

	def step(self):

		self.step_no += 1

	def __call__(self, function, *args, **kwargs):



		if 'attr' in kwargs:
			attr = kwargs['attr']
			kwargs.pop('attr', None)

		else:

			if type(function).__name__ == "method":
				attr = function.__name__

			elif type(function).__name__ == 'function':

				attr = str(function).split()[1]

			else:

				attr = type(function).__name__
				if attr == 'type':
					attr = str(function)[:-2].split('.')[-1]+'__init__'

		if attr == 'function':
			print(function, (type(function).__name__))

		if 'profiler_type' in kwargs:
			profiler_type = kwargs['profiler_type']
			kwargs.pop('profiler_type', None)
		else:
			profiler_type = 'multiple'
			
		self.found = False
		self.find_exists(attr, self.attrs)

		if not self.found:
			stack_copy = self.stack.copy()[::-1]
			self.ptr = self.attrs
			while len(stack_copy):
				to_pop = stack_copy.pop()
				
				found = False
				for process in self.ptr:
					if process['name'] == to_pop:
						self.ptr = process['children']
						found = True
						break
				if not found:
					self.ptr.append({'name':to_pop, 'time':0, 'count':0, 'children':[], 'profiler_type':profiler_type})#, 'count':0
					# print("Appended:",to_pop)
					for process in self.ptr:
						if process['name'] == to_pop:
							self.ptr = process['children']
							found = True
							break
			self.ptr.append({'name':attr, 'time':0, 'count':0, 'children':[], 'profiler_type':profiler_type})# , 'count':0

		self.stack.append(attr)

		before = time.time()

		if len(args) == 0 and len(kwargs) == 0:
			to_return = function()
		else:
			to_return = function(*args, **kwargs)

		after = time.time()

		self.stack.pop()

		self.update_value(attr, after-before, self.attrs)

		return to_return

	def reset_profiler(self):

		if os.path.exists(self.write_path):
			os.remove(self.write_path)

		if os.path.exists(self.config['dir']['Profiler']):

			folder = self.config['dir']['Profiler']
			
			for the_file in os.listdir(folder):
				file_path = os.path.join(folder, the_file)
				try:
					if os.path.isfile(file_path):
						os.unlink(file_path)
					elif os.path.isdir(file_path): 
						shutil.rmtree(file_path)
				except Exception as e:
					print(e)
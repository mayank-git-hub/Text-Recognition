import yaml
import os
import pickle
from .helper.read_yaml import read_yaml
from .helper.logger import Logger
from .helper.profiler import Profiler

log = Logger()

class PipelineManager():

	def __init__(self):

		self.config_file_d = read_yaml()
		for i in self.config_file_d['dir']:
			if not os.path.exists(self.config_file_d['dir'][i]):
				os.mkdir(self.config_file_d['dir'][i])

		self.config_file_r = read_yaml('configs/text_config.yaml')
		for i in self.config_file_r['dir']:
			if not os.path.exists(self.config_file_r['dir'][i]):
				os.mkdir(self.config_file_r['dir'][i])

		self.config_dataset = read_yaml('configs/dataset.yaml')

	#constructor loads yaml file containing hyperparameters

	#Working
	def train_d(self):
		self.profiler = Profiler(self.config_file_d)
		train_d(self.config_file_d, self.profiler)

	#Not Working
	def train_r(self):
		self.profiler = Profiler(self.config_file_r)
		train_r(self.config_file_r, self.profiler)

	# Not yet implemented

	# def train_rd(self, model_name):
	# 	train_rd(model_name, self.profiler)

	def prepare_metadata(self):
		self.profiler = Profiler(self.config_file_d)
		prepare_metadata(self.config_dataset, self.profiler)

	def test_d(self):
		self.profiler = Profiler(self.config_file_d)
		test_d(self.config_file_d, self.profiler)

	#Not Working
	def test_r(self):
		self.profiler = Profiler(self.config_file_r)
		test_r(self.config_file_r, self.profiler)

	#Not Working
	def test_rd(self):
		self.profiler = Profiler(self.config_file_d)
		test_rd(self.profiler)

	def test_one_d(self, path=None, out_path=None):
		self.profiler = Profiler(self.config_file_d)
		test_one_image_d(self.config_file_d, path, out_path, self.profiler)

	#Working
	def test_one_r(self, path, out_path):
		self.profiler = Profiler(self.config_file_r)
		test_one_image_r(self.config_file_r, path, out_path, self.profiler)

	#Not Working
	def test_one_rd(self, path=None, out_path=None):
		self.profiler = Profiler(self.config_file_d)
		test_one_image_rd(self.config_file_r, self.config_file_d, path, out_path, self.profiler)

	#Working
	def test_entire_folder_d(self, path='/home/mayank/Desktop/GitRepos/Text/Dataset/Passports_Dataset', out_path='/home/mayank/Desktop/GitRepos/Text/Dataset/Passports_Dataset_output'):
		self.profiler = Profiler(self.config_file_d)
		test_entire_folder_d(self.config_file_d, path, out_path, self.profiler)

	#Working
	def test_entire_folder_r(self, path=None, out_path=None):
		self.profiler = Profiler(self.config_file_r)
		if path == None:
			path = self.config_file_r['dir']['test_one']+'/images'
		if out_path == None:
			if not os.path.exists(self.config_file_r['dir']['test_one']+'/results'):
				os.mkdir(self.config_file_r['dir']['test_one']+'/results')
			out_path = self.config_file_r['dir']['test_one']+'/results'

		test_entire_folder_r(self.config_file_r, path, out_path, self.profiler)

	#Not Working
	def test_entire_folder_rd(self, path='/home/mayank/Desktop/GitRepos/Text/Dataset/Passports_Dataset', out_path='/home/mayank/Desktop/GitRepos/Text/Dataset/Passports_Dataset_output'):
		self.profiler = Profiler(self.config_file_d)
		test_entire_folder_rd(self.config_file_r, self.config_file_d, path, out_path, self.profiler)

	#Working
	def api(self):
		self.profiler = Profiler(self.config_file_d)
		api(self.profiler)

def train_d(config, profiler):

	from .Dlmodel.TrainTestD import TrainTestD

	driver = profiler(TrainTestD, config, 'train', profiler, profiler_type='once')
	success = profiler(driver.train_d, profiler_type='once')
	log.info(success)

	#object of Dlmodel created
	profiler.dump()
	profiler.plot()

def train_r(config, profiler):

	from .Dlmodel.TrainTestR import TrainTestR

	driver = profiler(TrainTestR, config, 'train', profiler, profiler_type='once')
	success = profiler(driver.train_r, profiler_type='once')
	log.info(success)

	#object of Dlmodel created

	profiler.dump()
	profiler.plot()

def test_d(config, profiler):

	from .Dlmodel.TrainTestD import TrainTestD

	driver = profiler(TrainTestD, config, 'test', profiler, profiler_type='once')#, profiler_type='single'
	profiler(driver.start_testing, profiler_type='once')
	profiler(driver.test_d, profiler_type='once')
		
	#object of dl_modle created in test mode


def test_r(config, profiler):

	from .Dlmodel.TrainTestR import TrainTestR

	driver = profiler(TrainTestR, config, 'test', profiler, profiler_type='once')#, profiler_type='single'
	profiler(driver.start_testing, profiler_type='once')
	profiler(driver.test_r, profiler_type='once')
		
	#object of dl_modle created in test mode

def test_one_image_r(config, path, out_path, profiler):

	from .Dlmodel.TestOneImageR import TestOneImageRClass

	driver = profiler(TestOneImageRClass, config, 'test_one', profiler, profiler_type='once')
	profiler(driver.test_one_image_r, path, out_path, profiler_type='once')
	#object of Dlmodel created in test_one mode


def test_entire_folder_r(config_file_r, path, out_path, profiler):

	from .Dlmodel.TestOneImageR import TestOneImageRClass

	driver = profiler(TestOneImageRClass, config_file_r, 'test_one', profiler, profiler_type='once')
	# print(path,out_path)
	images = [i for i in sorted(os.listdir(path)) if '.' != i[0]]
	# print(images)
	for image in images:
		print("Recognising text in image:",image)
		profiler(driver.test_one_image_r, path+'/'+image, out_path, profiler_type='once')
		

def test_one_image_rd(config_r, config_d, path, out_path, profiler):

	from .Dlmodel.TestOneImageRD import TestOneImageRDClass

	driver = profiler(TestOneImageRDClass, config_r, config_d, 'test_one', profiler, profiler_type='once')
	contours, text = profiler(driver.test_one_image_rd, path, out_path, profiler_type='once')

	print(contours.shape, text)
	return contours, text

	#object of Dlmodel created in test_one mode

def gen_rd(model, path, out, out_label):

	all_files = os.listdir(path)

	#all_files is a list with all the files(could be a directory) in the directory 'path'
	#creates input and output directories, and checks that image format is correct

	for file_i in all_files:

		if file_i.split('.')[-1].lower() in ['jpeg', 'png', 'jpg']: 
			log.info(path+'/'+file_i)
			if not (os.path.exists(out+'/'+file_i) and os.path.exists(out_label+'/'+'.'.join(file_i.split('.')[:-1])+'.pkl')):
				contours, text = model.test_one_image_rd(path+'/'+file_i, out+'/'+file_i)
				with open(out_label+'/'+'.'.join(file_i.split('.')[:-1])+'.pkl', 'wb') as f:
					pickle.dump([contours, text], f)


		elif os.path.isdir(path+'/'+file_i):
			if not os.path.exists(out+'/'+file_i):
				os.mkdir(out+'/'+file_i)
			if not os.path.exists(out_label+'/'+file_i):
				os.mkdir(out_label+'/'+file_i)

			gen_rd(model, path+'/'+file_i, out+'/'+file_i, out_label+'/'+file_i)
			#if the file in the list all_files is a directory, call the function again

def test_entire_folder_rd(config_r, config_d, path, out_path, profiler):

	#calls object of api_class

	#creates output directory if it doesn't exist, calls gen
	out_label = out_path+'_label'
	log.info(path, out_path, out_label)

	from .Dlmodel.TestOneImageRD import TestOneImageRDClass

	if not os.path.exists(out_path):
		os.mkdir(out_path)

	if not os.path.exists(out_label):
		os.mkdir(out_label)

	driver = profiler(TestOneImageRDClass, config_r, config_d, 'test_one', profiler, profiler_type='once')
	profiler(gen_rd, driver, path, out_path, out_label, profiler_type='once')

def test_one_image_d(config, path, out_path, profiler):

	from .Dlmodel.TestOneImageD import TestOneImageDClass

	driver = profiler(TestOneImageDClass, config, 'test_one', profiler, profiler_type='once')
	profiler(driver.test_one_image_d, path, out_path, profiler_type='once')

	#object of Dlmodel created in test_one mode
 
def gen_d(model, path, out):

	all_files = os.listdir(path)

	#all_files is a list with all the files(could be a directory) in the directory 'path'
	#creates input and output directories, and checks that image format is correct

	for file_i in all_files:

		if file_i.split('.')[-1].lower() in ['jpeg', 'png', 'jpg', 'gif']: 
			log.info(path+'/'+file_i)
			if not os.path.exists(out+'/'+file_i):
				model.test_one_image_d(path+'/'+file_i, out+'/'+file_i)

		elif os.path.isdir(path+'/'+file_i):
			if not os.path.exists(out+'/'+file_i):
				os.mkdir(out+'/'+file_i)
			gen_d(model, path+'/'+file_i, out+'/'+file_i)
			#if the file in the list all_files is a directory, call the function again

def test_entire_folder_d(config, path, out_path, profiler):

	#calls object of api_class

	#creates output directory if it doesn't exist, calls gen

	print(path, out_path)

	from .Dlmodel.TestOneImageD import TestOneImageDClass

	if not os.path.exists(out_path):
		os.mkdir(out_path)
	driver = profiler(TestOneImageDClass, config, 'test_one', profiler=profiler, profiler_type='once')
	profiler(gen_d, driver, path, out_path, profiler_type='once')

def prepare_metadata(config, profiler):

	#reads dataset name from yaml file, and uses corresponding metadata
	datasets = set(config['dataset_train']+config['dataset_test'])
	
	for d_name in datasets:

		if d_name == 'COCO':
			from .prepare_metadata.prepare_metadata import MetaCoco as meta

		elif d_name == 'IC13':
			from .prepare_metadata.prepare_metadata import MetaIC13 as meta

		elif d_name == 'IC15':
			from .prepare_metadata.prepare_metadata import MetaIC15 as meta

		elif d_name == 'SYNTH':
			from .prepare_metadata.prepare_metadata import MetaSynth as meta

		elif d_name == 'OWN':
			from .prepare_metadata.prepare_metadata import MetaOwn as meta

		elif d_name == 'ART':
			from .prepare_metadata.prepare_metadata import MetaArtificial as meta

		log.info("Metadata for:", d_name)
		wrapper = profiler(meta, config)
		profiler(wrapper.create_annot)

	#object of prepare_metadata created
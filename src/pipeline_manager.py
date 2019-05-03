import os
import pickle

from .helper.logger import Logger
from .helper.profiler import Profiler
from .helper.read_yaml import read_yaml

log = Logger()


class PipelineManager:
	"""
	A Class to outline the functions and decide which config file to send to which function
	"""

	def __init__(self):

		"""
		Function to get 3 yaml files
			self.config_file_d: Contains the dynamic and static attributes related to Detection Model
			self.config_file_r: Contains the dynamic and static attributes related to Recognition Model
			self.config_dataset: Contains static attributes related to Dataset
		The Functions of the class are self explanatory. Hence no commenting needed.
		"""

		self.config_file_d = read_yaml()
		for i in self.config_file_d['dir']:
			if not os.path.exists(self.config_file_d['dir'][i]):
				os.mkdir(self.config_file_d['dir'][i])

		self.config_file_r = read_yaml('configs/text_config.yaml')
		for i in self.config_file_r['dir']:
			if not os.path.exists(self.config_file_r['dir'][i]):
				os.mkdir(self.config_file_r['dir'][i])

		self.config_dataset = read_yaml('configs/dataset.yaml')

	def train_d(self):

		self.profiler = Profiler(self.config_file_d)
		train_d(self.config_file_d, self.profiler)

	def train_r(self):

		self.profiler = Profiler(self.config_file_r)
		train_r(self.config_file_r, self.profiler)

	def prepare_metadata(self):

		self.profiler = Profiler(self.config_file_d)
		prepare_metadata(self.config_dataset, self.profiler)

	def test_d(self):

		self.profiler = Profiler(self.config_file_d)
		test_d(self.config_file_d, self.profiler)

	def test_r(self):

		self.profiler = Profiler(self.config_file_r)
		test_r(self.config_file_r, self.profiler)

	def test_one_d(self, path=None, out_path=None):

		self.profiler = Profiler(self.config_file_d)
		test_one_d(self.config_file_d, path, out_path, self.profiler)

	def test_one_r(self, path, out_path):

		self.profiler = Profiler(self.config_file_r)
		test_one_r(self.config_file_r, path, out_path, self.profiler)

	def test_one_rd(self, path=None, out_path=None):

		self.profiler = Profiler(self.config_file_d)
		test_one_rd(self.config_file_r, self.config_file_d, path, out_path, self.profiler)

	def test_entire_folder_d(self, path, out_path):

		self.profiler = Profiler(self.config_file_d)
		test_entire_folder_d(self.config_file_d, path, out_path, self.profiler)

	def test_entire_folder_r(self, path, out_path):

		self.profiler = Profiler(self.config_file_r)
		test_entire_folder_r(self.config_file_r, path, out_path, self.profiler)

	def test_entire_folder_rd(self, path, out_path):

		self.profiler = Profiler(self.config_file_d)
		test_entire_folder_rd(self.config_file_r, self.config_file_d, path, out_path, self.profiler)


def train_d(config, profiler):

	"""
	Have a look at main.py for description

	:param config: dynamic variables taken from the folder configs
	:param profiler: A class to generate statistics of time usage of each function
	"""

	from .Dlmodel.TrainTestD import TrainTestD

	driver = profiler(TrainTestD, config, 'train', profiler, profiler_type='once')
	success = profiler(driver.train_d, profiler_type='once')
	log.info(success)

	profiler.dump()
	profiler.plot()


def train_r(config, profiler):
	"""
	Have a look at main.py for description

	:param config: dynamic variables taken from the folder configs
	:param profiler: A class to generate statistics of time usage of each function
	"""

	from .Dlmodel.TrainTestR import TrainTestR

	driver = profiler(TrainTestR, config, 'train', profiler, profiler_type='once')
	success = profiler(driver.train_r, profiler_type='once')
	log.info(success)

	profiler.dump()
	profiler.plot()


def test_d(config, profiler):
	"""
	Have a look at main.py for description

	:param config: dynamic variables taken from the folder configs
	:param profiler: A class to generate statistics of time usage of each function
	"""

	from .Dlmodel.TrainTestD import TrainTestD

	driver = profiler(TrainTestD, config, 'test', profiler, profiler_type='once')  # , profiler_type='single'
	profiler(driver.start_testing, profiler_type='once')
	profiler(driver.test_d, profiler_type='once')


def test_r(config, profiler):
	"""
	Have a look at main.py for description

	:param config: dynamic variables taken from the folder configs
	:param profiler: A class to generate statistics of time usage of each function
	"""

	from .Dlmodel.TrainTestR import TrainTestR

	driver = profiler(TrainTestR, config, 'test', profiler, profiler_type='once')  # , profiler_type='single'
	profiler(driver.start_testing, profiler_type='once')
	profiler(driver.test_r, profiler_type='once')


def test_one_r(config, path, out_path, profiler):
	"""
	Have a look at main.py for description

	:param config: dynamic variables taken from the folder configs
	:param path: Input Image path
	:param out_path: Output Image Path with filename as text_predicted
	:param profiler: A class to generate statistics of time usage of each function
	"""

	from .Dlmodel.TestOneImageR import TestOneImageRClass

	driver = profiler(TestOneImageRClass, config, 'test_one', profiler, profiler_type='once')
	profiler(driver.test_one_image_r, path, out_path, profiler_type='once')


def test_entire_folder_r(config_file_r, path, out_path, profiler):
	"""
	Have a look at main.py for description

	:param config_file_r: dynamic variables taken from the folder configs
	:param path: Input Folder path
	:param out_path: Output Folder Path with filename as text_predicted
	:param profiler: A class to generate statistics of time usage of each function
	"""

	from .Dlmodel.TestOneImageR import TestOneImageRClass

	driver = profiler(TestOneImageRClass, config_file_r, 'test_one', profiler, profiler_type='once')
	images = [i for i in sorted(os.listdir(path)) if '.' != i[0]]

	for image in images:
		print("Recognising text in image:", image)
		profiler(driver.test_one_image_r, path + '/' + image, out_path, profiler_type='once')


def test_one_rd(config_r, config_d, path, out_path, profiler):
	"""
	Have a look at main.py for description

	:param config_r: dynamic variables taken from the folder configs(recognition)
	:param config_d: dynamic variables taken from the folder configs(detection)
	:param path: Input Image path
	:param out_path: Output Image Path with filename as text_predicted
	:param profiler: A class to generate statistics of time usage of each function
	"""

	from .Dlmodel.TestOneImageRD import TestOneImageRDClass

	driver = profiler(TestOneImageRDClass, config_r, config_d, 'test_one', profiler, profiler_type='once')
	contours, text = profiler(driver.test_one_image_rd, path, out_path, profiler_type='once')

	import pickle
	pickle.dump([contours, text], open(out_path + '/output.pkl', 'wb'))

	return contours, text


def test_entire_folder_rd(config_r, config_d, ipath, opath, profiler):
	"""
	Have a look at main.py for description

	:param config_r: dynamic variables taken from the folder configs(recognition)
	:param config_d: dynamic variables taken from the folder configs(detection)
	:param ipath: Input Image path
	:param opath: Output Image Path with filename as text_predicted
	:param profiler: A class to generate statistics of time usage of each function
	"""

	def gen_rd(model, path, out, out_label):

		"""
		A Recursive function to be called to generate output which is described in main.py

		:param model: model object which contains functions to detect and recognise
		:param path: Current Folder path
		:param out: Current Output path
		:param out_label: Current Label path
		"""

		all_files = os.listdir(path)

		# all_files is a list with all the files(could be a directory) in the directory 'path'
		# creates input and output directories, and checks that image format is correct

		for file_i in all_files:

			if file_i.split('.')[-1].lower() in ['jpeg', 'png', 'jpg']:
				log.info(path + '/' + file_i)
				if not (os.path.exists(out + '/' + file_i) and os.path.exists(
						out_label + '/' + '.'.join(file_i.split('.')[:-1]) + '.pkl')):
					contours, text = model.test_one_image_rd(path + '/' + file_i, out + '/' + file_i)
					with open(out_label + '/' + '.'.join(file_i.split('.')[:-1]) + '.pkl', 'wb') as f:
						pickle.dump([contours, text], f)

			elif os.path.isdir(path + '/' + file_i):
				if not os.path.exists(out + '/' + file_i):
					os.mkdir(out + '/' + file_i)
				if not os.path.exists(out_label + '/' + file_i):
					os.mkdir(out_label + '/' + file_i)

				gen_rd(model, path + '/' + file_i, out + '/' + file_i, out_label + '/' + file_i)

			# if the file in the list all_files is a directory, call the function again

	# creates output directory if it doesn't exist, calls gen_rd

	first_out_label = opath + '_label'
	log.info(ipath, opath, first_out_label)

	from .Dlmodel.TestOneImageRD import TestOneImageRDClass

	if not os.path.exists(opath):
		os.mkdir(opath)

	if not os.path.exists(first_out_label):
		os.mkdir(first_out_label)

	driver = profiler(TestOneImageRDClass, config_r, config_d, 'test_one', profiler, profiler_type='once')
	profiler(gen_rd, driver, ipath, opath, first_out_label, profiler_type='once')


def test_one_d(config, path, out_path, profiler):
	"""
	Have a look at main.py for description

	:param config: dynamic variables taken from the folder configs
	:param path: Input Image path
	:param out_path: Output Image Path with filename as text_predicted
	:param profiler: A class to generate statistics of time usage of each function
	"""

	from .Dlmodel.TestOneImageD import TestOneImageDClass

	driver = profiler(TestOneImageDClass, config, 'test_one', profiler, profiler_type='once')
	profiler(driver.test_one_image_d, path, out_path, profiler_type='once')


def test_entire_folder_d(config, ipath, opath, profiler):
	"""
	Have a look at main.py for description

	:param config: dynamic variables taken from the folder configs
	:param ipath: Input Image path
	:param opath: Output Image Path with filename as text_predicted
	:param profiler: A class to generate statistics of time usage of each function
	"""

	def gen_d(model, path, out):

		"""
		A Recursive function to be called to generate output which is described in main.py

		:param model: model object which contains functions to detect and recognise
		:param path: Current Folder path
		:param out: Current Output path
		"""

		all_files = os.listdir(path)

		# all_files is a list with all the files(could be a directory) in the directory 'path'
		# creates input and output directories, and checks that image format is correct

		for file_i in all_files:

			if file_i.split('.')[-1].lower() in ['jpeg', 'png', 'jpg', 'gif']:
				log.info(path + '/' + file_i)
				if not os.path.exists(out + '/' + file_i):
					model.test_one_image_d(path + '/' + file_i, out + '/' + file_i)

			elif os.path.isdir(path + '/' + file_i):
				if not os.path.exists(out + '/' + file_i):
					os.mkdir(out + '/' + file_i)
				gen_d(model, path + '/' + file_i, out + '/' + file_i)

			# if the file in the list all_files is a directory, call the function again

	# creates output directory if it doesn't exist, calls gen

	print(ipath, opath)

	from .Dlmodel.TestOneImageD import TestOneImageDClass

	if not os.path.exists(opath):
		os.mkdir(opath)
	driver = profiler(TestOneImageDClass, config, 'test_one', profiler=profiler, profiler_type='once')
	profiler(gen_d, driver, ipath, opath, profiler_type='once')


def prepare_metadata(config, profiler):
	"""
	Have a look at main.py for description

	:param config: dynamic variables taken from the folder configs
	:param profiler: A class to generate statistics of time usage of each function
	"""

	datasets = set(config['dataset_train'] + config['dataset_test'])

	for d_name in datasets:

		if d_name == 'COCO':
			from .prepare_metadata.prepare_metadata import MetaCoco as Meta

		elif d_name == 'IC13':
			from .prepare_metadata.prepare_metadata import MetaIC13 as Meta

		elif d_name == 'IC15':
			from .prepare_metadata.prepare_metadata import MetaIC15 as Meta

		elif d_name == 'SYNTH':
			from .prepare_metadata.prepare_metadata import MetaSynth as Meta

		elif d_name == 'OWN':
			from .prepare_metadata.prepare_metadata import MetaOwn as Meta

		elif d_name == 'ART':
			from .prepare_metadata.prepare_metadata import MetaArtificial as Meta
		else:
			log.info('Dataset: ', d_name, 'Not Implemented')
			return False

		log.info("Metadata for:", d_name)
		wrapper = profiler(Meta, config)
		profiler(wrapper.create_annot)

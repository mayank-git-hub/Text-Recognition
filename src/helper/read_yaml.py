import yaml
def read_yaml(path = 'configs/config.yaml'):
	with open(path, 'r') as stream:
		try:
			with open('configs/dataset.yaml') as fixed_stream:
				z = {**yaml.load(stream), **yaml.load(fixed_stream)}
				return z
		except yaml.YAMLError as exc:
			return exc
#read yaml files that defines hyperparameters and the location of data
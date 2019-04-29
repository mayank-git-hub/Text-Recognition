import click
from src.pipeline_manager import PipelineManager
from src.helper.logger import Logger

@click.group()
def main():
	pass

#Click library to provide interface on terminal
#eg. python main.py train -m ResNet_UNet
#Object of PipeLine manager and logger created

@main.command()
def train_d():
	# Training Detection
	pipeline_manager.train_d()

@main.command()
def train_r():
	# Training Recognition
	pipeline_manager.train_r()

@main.command()
def test_d():
	# Testing Pretrained Detection, specify the path for pretrained model in configs/config.yaml
	pipeline_manager.test_d()

@main.command()
def test_r():
	# Testing Pretrained Recognition(Only on cropped images), specify the path for pretrained model in configs/text_config.yaml
	pipeline_manager.test_r()

@main.command()
def test_rd():
	# Testing Pretrained Recognition and Detection Model on Entire Images. Specify the path for pretrained model in configs/text_config.yaml, config.yaml
	pipeline_manager.test_rd()

@main.command()
@click.option('-p', '--path', help='Image Path' , required=True)
@click.option('-o', '--out_path', help='Save Path' , required=True)
def test_one_d(path, out_path):
	# Testing One image for detection, provided the path and saving the output in out_path. Speicify the path for pretrained model in configs/config.yaml
	pipeline_manager.test_one_d(path, out_path)

@main.command()
@click.option('-i', '--ipath', help='Input Path', required=True)
@click.option('-o', '--opath', help='Output Path', required=True)

def test_one_r(ipath, opath):
	# Testing One image for recognition(Only Cropped Images), provided the path and printing the output on the screen. Speicify the path for pretrained model in configs/text_config.yaml
	pipeline_manager.test_one_r(ipath, opath)

@main.command()
@click.option('-p', '--path', help='Image Path' , required=True)
@click.option('-o', '--out_path', help='Save Path' , required=True)
def test_one_rd(path, out_path):
	# Testing One image for detection and recognition, provided the path and saves the output of detection in the path specified and returns the contour, text. Speicify the path for pretrained model in configs/config.yaml, text_config.yaml
	pipeline_manager.test_one_rd(path, out_path)

@main.command()
@click.option('-p', '--path', help='Image Path' , required=False)
@click.option('-o', '--out_path', help='Save Path' , required=False)
def test_entire_folder_d(path, out_path):

	if path == None and out_path == None:
		pipeline_manager.test_entire_folder_d()
	else:
		pipeline_manager.test_entire_folder_d(path, out_path)

@main.command()
@click.option('-p', '--path', help='Image Path' , required=False)
@click.option('-o', '--out_path', help='Save Path' , required=False)
def test_entire_folder_r(path, out_path):

	if path == None and out_path == None:
		pipeline_manager.test_entire_folder_r()
	else:
		pipeline_manager.test_entire_folder_r(path, out_path)

@main.command()
@click.option('-p', '--path', help='Image Path' , required=False)
@click.option('-o', '--out_path', help='Save Path' , required=False)
def test_entire_folder_rd(path, out_path):

	if path == None and out_path == None:
		pipeline_manager.test_entire_folder_rd()
	else:
		pipeline_manager.test_entire_folder_rd(path, out_path)

#For test_one_image and test_entire_folder, image input and output directories have to be specified

@main.command()
@click.option('-p', '--predicted', help='Predicted' , required=True)
@click.option('-t', '--target', help='Target' , required=True)
@click.option('-th', '--threshold', help='Threshold' , required=True)
@click.option('-text', '--text', help='Text' , required=False)
def fscore(predicted, target, threshold, text):

	from src.helper.utils import get_f_score
	if text == 'True':
		get_f_score(predicted, target, float(threshold), True)
	else:
		get_f_score(predicted, target, float(threshold), False)

@main.command()
def prepare_metadata():
	pipeline_manager.prepare_metadata()

# @main.command()
# @click.option('-m', '--model', help='ResNet_UNet' , required=False)
# def train_rd(model):
# 	pipeline_manager.train_rd(model)

if __name__ == "__main__":

	pipeline_manager = PipelineManager()
	log = Logger()
	log.first()
	main()

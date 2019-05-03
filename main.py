import click

from src.helper.logger import Logger
from src.pipeline_manager import PipelineManager


@click.group()
def main():
	pass


@main.command()
def train_d():
	"""
	Used to train the detection part of the TD&R closely based on Pixel Link
	The detection is done using instance segmentation, model UNET-RESNET
	The data set used is SynthText and our Custom Dataset ToDo (which we will be releasing soon)
	The data sets contain the bounding box of every "text word" - ToDo (Word is vague to special characters)
	"""

	pipeline_manager.train_d()


@main.command()
def train_r():
	"""
	Used to train the recognition part of the TD&R closely based on Pixel Link
	The recognition is done using CRNN, model - Custom Convolution + Bi-LSTM
	"""

	pipeline_manager.train_r()


@main.command()
def test_d():
	"""
	Testing Pre-trained Detection model on a folder structure as mentioned in config.yaml
	Can Run on images of varying size and extension
	"""

	pipeline_manager.test_d()


@main.command()
def test_r():
	"""
	Testing Pre-trained Recognition model on a folder structure as mentioned in config.yaml
	Can Run on images which have closely cropped words
	"""

	pipeline_manager.test_r()


@main.command()
def test_rd():
	"""
	Testing Pretrained Recognition and Detection Model on a folder structure as mentioned in config.yaml
	Can Run on images of varying size and extension
	"""

	pipeline_manager.test_rd()


@main.command()
@click.option('-p', '--ipath', help='Image Path', required=True)
@click.option('-o', '--opath', help='Save Path', required=True)
def test_one_d(ipath, opath):
	"""
	Testing One image for detection, provided the input and output path
	:input:
		:param ipath: Path to the Image on which detection model is to be run
		:param opath: Folder Path where the output is stored which contains
		                    Original Image with contour drawn on it
		                    Blank Image with contours filled with various colors drawn on it
		                    8 Images which contain Links in all directions
		                    1 Image showing pixel where semantic segmentation was positive and Link was negative
		                    Image should Target (ToDo elaborate)
		                    Image showing continuous semantic segmentation values
	"""

	pipeline_manager.test_one_d(ipath, opath)


@main.command()
@click.option('-i', '--ipath', help='Input Path', required=True)
@click.option('-o', '--opath', help='Output Path', required=True)
def test_one_r(ipath, opath):
	"""
	Testing One image for recognition(Only Cropped Images)

	:input:
		:param ipath: Path to the Word Cropped Image on which Recognition model is to be run
		:param opath: Output Path to the Word Cropped Image whose path name is what was predicted
	"""

	pipeline_manager.test_one_r(ipath, opath)


@main.command()
@click.option('-p', '--ipath', help='Image Path', required=True)
@click.option('-o', '--opath', help='Save Path', required=True)
def test_one_rd(ipath, opath):
	"""
	Testing One Image(No constraints) for recognition and detection

	:input:
		:param ipath: Path to the Image on which Recognition and Detection model is to be run
		:param opath: Folder Path where the output is stored which contains
	                        Original Image with contour drawn on it
		                    Blank Image with contours filled with various colors drawn on it
		                    8 Images which contain Links in all directions
		                    1 Image showing pixel where semantic segmentation was positive and Link was negative
		                    Image should Target (ToDo elaborate)
		                    Image showing continuous semantic segmentation values
		                    output.pkl which contains all the contours and text in format mentioned in ReadMe.md
	"""

	pipeline_manager.test_one_rd(ipath, opath)


@main.command()
@click.option('-p', '--ipath', help='Image Path', required=True)
@click.option('-o', '--opath', help='Save Path', required=True)
def test_entire_folder_d(ipath, opath):
	"""
	Testing Entire Folder Recursively on Images(No constraints) for recognition and detection

	:input:
		:param ipath: Path to the Image on which Detection model is to be run
		:param opath: Folder Path where the output is stored exactly in structure the ipath folder was
	                  and output similar to test_one_d             
	"""

	pipeline_manager.test_entire_folder_d(ipath, opath)


@main.command()
@click.option('-p', '--ipath', help='Image Path', required=True)
@click.option('-o', '--opath', help='Save Path', required=True)
def test_entire_folder_r(ipath, opath):
	# ToDo @mithilesh comment the function

	pipeline_manager.test_entire_folder_r(ipath, opath)


@main.command()
@click.option('-p', '--ipath', help='Image Path', required=True)
@click.option('-o', '--opath', help='Save Path', required=True)
def test_entire_folder_rd(ipath, opath):
	"""
	Testing Entire Folder for Images(No constraints) for recognition and detection

	:input:
		:param ipath: Path to the Folder to search for the Images on which Recognition and Detection model is to be run
		:param opath: Folder Path where the output is stored which contains
	                        Original Image with contour drawn on it
		                    Blank Image with contours filled with various colors drawn on it
		                    8 Images which contain Links in all directions
		                    1 Image showing pixel where semantic segmentation was positive and Link was negative
		                    Image should Target (ToDo elaborate)
		                    Image showing continuous semantic segmentation values
		                    output.pkl which contains all the contours and text in format mentioned in ReadMe.md
		              Folder Path + "_label" where the output is stored which contains
		                    output.pkl which contains all the contours and text in format mentioned in ReadMe.md
	"""

	pipeline_manager.test_entire_folder_rd(ipath, opath)


@main.command()
@click.option('-p', '--predicted', help='Predicted', required=True)
@click.option('-t', '--target', help='Target', required=True)
@click.option('-th', '--threshold', help='Threshold', required=True)
@click.option('-text', '--text', help='Text', required=False)
def fscore(predicted, target, threshold, text):
	"""
	A Function to calculate the Fscore of the entire folder with respect to a folder containing original labels
	:input:
		:param predicted: Folder path to where the predicted labels are
		:param target: Folder path to where the target labels are
		:param threshold: Threshold to set on IOU for classifying as positive or negative
		:param text: If text is available this flag can be used to classify a bbox as positive if R&D both are correct
	"""
	
	from src.helper.utils import get_f_score
	if text == 'True':
		get_f_score(predicted, target, float(threshold), True)
	else:
		get_f_score(predicted, target, float(threshold), False)


@main.command()
def prepare_metadata():
	"""
	Pre-processing all the data sets mentioned in dataset.yaml in the form mentioned in Readme.md
	"""

	pipeline_manager.prepare_metadata()


if __name__ == "__main__":
	"""
	Common Initialisation to every task
	"""

	pipeline_manager = PipelineManager()
	log = Logger()
	log.first()
	main()

# Pytorch Implementation of [Pixel-LINK](https://arxiv.org/pdf/1801.01315.pdf)

## Instructions to run the code

### Setting up the dataset

	1. In the configs/dataset.yaml file add your dataset in the following format under the field metadata

		1. <Name of the dataset>
			1. dir:<Path-to-Dataset-Folder> 
			2. image: <Path-to-Dataset-Folder>/Images
			3. label: <Path-to-Dataset-Folder>/Labels
			4. meta: <Path-to-Dataset-Folder>/Meta
			5. contour_length_thresh_min: <Contours with length less than this are excluded from training and testing>
			6. contour_area_thresh_min: <Contours with area less than this are excluded from training and testing>
			7. segmentation_thresh: <Confidence value over which pixel is classified as positive>
    		8. link_thresh: <Confidence value over which link is classified as positive>
    		9. cal_avg: <If True: Padding with Average of the image, else: Padding with Zeros>
    		10. split_type: <% of Training Images which is randomly picked from the dataset, remaining is used for validation>

	2. Put all your images in the *<Path-to-Dataset-Folder>/Images* folder

	3. Create Labels in the format - 
		1. Contours = List of all bounding box(dtype=np.float32, shape=[4, 1, 2](4, 2 for four co-ordinates with two dimensions))
		2. Text = List of all strings which have text corresponding to every Contour
		3. Labels corresponding to every image would have the name <image-name.extension-of-image.pkl>. It will be a pickle dump of the list [Contours, Text]

	4. Save all the labels for the images in the folder *<Path-to-Dataset-Folder>/Labels*

	5. Create the folder *<Path-to-Dataset-Folder>/Meta*

	6. In the configs/dataset.yaml file put your dataset name in the field *dataset_train* and *dataset_test*

	7. Run python main.py prepare_metadata

### Training your own model(Detection)
	
	1. The configs/config.yaml contains all the hyper-parameters for training the detection model.
	2. After your dataset and config file is in place run the command `python main.py train_d`

### Testing your own model(Detection)
	
	1. In the configs/config.yaml in the field "PreTrained_model" change the value of the field "check" to True
	2. Configure the path of the model in the field "PreTrained_Model"
	3. After your dataset and config file is in place run the command `python main.py test_d`

### Generate Visual Results on a single image

	1. In the configs/config.yaml in the field "PreTrained_model" change the value of the field "check" to True
	2. Configure the path of the model in the field "PreTrained_Model"
	3. Run the command `python main.py test_one_d -p <path-to-test-image> -o <path-to-folder-output>`

### Generate Visual Results on an entire folder

	1. In the configs/config.yaml in the field "PreTrained_model" change the value of the field "check" to True
	2. Configure the path of the model in the field "PreTrained_Model"
	3. Run the command `python main.py test_entire_folder_d -p <path-to-test-folder> -o <path-to-output-folder>`

This repository is currently into active development. Do raise issues and we will solve them as soon as possible.

Public Board on Trello - https://trello.com/b/V26dOOOB

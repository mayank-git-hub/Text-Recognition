from nltk.corpus import words
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from io import BytesIO
import time
import cv2
import os
import pickle

class MetaArtificial():

	def __init__(self, config):

		self.config = config
		self.seed()
		self.split_type = self.config['metadata']['ART']['split_type']

		self.all_words = words.words()

		self.all_hindi_char = ['\u0904', '\u0905', '\u0906', '\u0907',
						  '\u0908', '\u0909', '\u090A', '\u090B', 
						  '\u090C', '\u090D', '\u090E', '\u090F', 
						  '\u0910', '\u0911', '\u0912', '\u0913', 
						  '\u0914', '\u0915', '\u0916', '\u0917', 
						  '\u0918', '\u0919', '\u091A', '\u091B', 
						  '\u091C', '\u091D', '\u091E', '\u091F',
						  '\u0920', '\u0921', '\u0922', '\u0923',
						  '\u0924', '\u0925', '\u0926', '\u0927',
						  '\u0928', '\u0929', '\u092A', '\u092B',
						  '\u092C', '\u092D', '\u092E', '\u092F',
						  '\u0930', '\u0931', '\u0932', '\u0933',
						  '\u0934', '\u0935', '\u0936', '\u0937',
						  '\u0938', '\u0939', '\u0950', '\u0958',
						  '\u0959', '\u095A', '\u095B', '\u095C',
						  '\u095D', '\u095E', '\u095F', '\u0960',
						  '\u0961', '\u0966', '\u0967', '\u0968',
						  '\u0969', '\u096A', '\u096B', '\u096C',
						  '\u096D', '\u096E', '\u096F', '\u0972',
						  '\u0973', '\u0974', '\u0975', '\u0976',
						  '\u0977',]

		self.all_hindi_matra =['\u0900', '\u0901', '\u0902', '\u0903',
						  '\u093a', '\u093b', '\u093c', '\u093d',
						  '\u093e', '\u093f', '\u0940', '\u0941',
						  '\u0942', '\u0943', '\u0944', '\u0945',
						  '\u0946', '\u0947', '\u0948', '\u0949',
						  '\u094A', '\u094B', '\u094C', '\u094D',
						  '\u094E', '\u094F', '\u0953', '\u0954',
						  '\u0955', '\u0956', '\u0957', '\u0962',
						  '\u0963',]

		self.english_alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
		
		self.english_symbols = ',./<>:;{[]+=?|}#!@$%^&*()_-'

		self.prob_add_matra = 0.2

		self.prob_add_symbol = 0.2

		self.transparent_mean = 0.8
		
		self.transparent_gaussian = 0.06
		
		self.prob_dep = 0.5

		self.hindi_prob = 0.1

		self.symbol_word = 0.05

		self.total_attempt = 5

		self.font_dir_name='/home/Common/Mayank/Text/Segmentation/Dataset/ART/fonts_for_text'

		self.font_dir_name_hindi='/home/Common/Mayank/Text/Segmentation/Dataset/ART/fonts_for_text_hindi'

		self.probability_dist = np.array([0.1, 0.6, 2.6, 5.2, 8.5, 12.2, 14, 14, 12.6, 10.1, 7.5, 5.2, 3.2, 2, 1, 0.6, 0.3, 0.2, 0.1, 0.1])
		self.probability_dist = self.probability_dist/np.sum(self.probability_dist)

		list_of_files = self.get_list_of_files(self.font_dir_name)
		list_of_files_hindi = self.get_list_of_files(self.font_dir_name_hindi)

		self.all_fonts = []
		for i in range(len(list_of_files)):
			with open(list_of_files[i],"rb") as f:
				font_bytes=f.read()
				self.all_fonts.append(font_bytes)

		self.all_fonts_hindi = []
		for i in range(len(list_of_files_hindi)):
			with open(list_of_files_hindi[i],"rb") as f:
				font_bytes=f.read()
				self.all_fonts_hindi.append(font_bytes)

		self.image_net_location = "/media/mayank/0b40607e-7efc-4216-b12f-8bb86facfaed/Image_Net/ILSVRC/Data/CLS-LOC/test/"
		self.image_save_location = '/home/Common/Mayank/Text/Segmentation/Dataset/ART/Images/'
		self.label_save_location = '/home/Common/Mayank/Text/Segmentation/Dataset/ART/Labels/'

	def get_all_names_refresh(self):
	
		IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif']
		paths_of_image=[]
		
		for image in os.listdir(self.image_net_location):
			if(((image).split('.')[-1]).lower() in IMG_EXTENSIONS):
				paths_of_image.append(image)

		paths_of_image = np.array(paths_of_image)
		np.random.shuffle(paths_of_image)
		
		return paths_of_image.tolist()

	def transparent(self, x, mean, gaussian):

		alpha = np.random.normal(loc = mean, scale = gaussian, size=x.shape[0]*x.shape[1]*x.shape[2]).reshape([x.shape[0], x.shape[1], x.shape[2]])
		x = 255*alpha + x*(1 - alpha)
		return x

	def seed(self):

		np.random.seed(self.config['seed'])

	def generate_lexicon_dependent(self):

		x = np.random.randint(len(self.all_words))
		return self.all_words[x]

	def generate_hindi(self, length):

		word = ''

		for i in range(length):
			word += self.all_hindi_char[np.random.randint(len(self.all_hindi_char))]
			if np.random.choice([0, 1], p=[self.prob_add_matra, 1- self.prob_add_matra]) == 0:
				word += self.all_hindi_matra[np.random.randint(len(self.all_hindi_matra))]
		# print(word)
		return word

	def generate_english(self, length):

		word = ''

		if np.random.choice([0, 1], p=[self.symbol_word, 1 - self.symbol_word]) == 0:
			symbol_length = [10, 0, 0, 0, 0, 0]+[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
			symbol_length = np.array(symbol_length).astype(np.float32)
			symbol_length /= np.sum(symbol_length)
			length = np.random.choice(np.arange(len(symbol_length))+1, p=symbol_length)
			for i in range(length):
				word += self.english_symbols[np.random.randint(len(self.english_symbols))]
		else:

			for i in range(length):
				word += self.english_alpha[np.random.randint(len(self.english_alpha))]				

		return word

	def generate_lexicon_free(self):

		
		length = np.random.choice(np.arange(1, 1+self.probability_dist.shape[0]), p=self.probability_dist)

		if np.random.choice([0, 1], p=[self.hindi_prob, 1- self.hindi_prob]) == 0:
			return self.generate_hindi(length),"hindi"

		return self.generate_english(length),"english"

	def gen_word(self):

		if np.random.choice([0, 1], p=[self.prob_dep, 1- self.prob_dep]) == 0:
			# print("generate_lexicon_dependent")
			return self.generate_lexicon_dependent(),"english"
		else:
			# print("generate_lexicon_free")
			return self.generate_lexicon_free()

	def which_overlap(self, coordinates, bounding_box):

		for i in range(len(bounding_box)):
			union_val,intersection_val=self.intersection_union(bounding_box[i], coordinates)
			if intersection_val!=0:
				return bounding_box[i]
		return None
 
	def inside_point(self, point, rect):

		# point is a list (x, y)
		# rect is a contour with shape [4, 2]

		rect = rect.reshape([4, 1, 2]).astype(np.int64)

		dist = cv2.pointPolygonTest(rect,(point[0], point[1]),True)

		if dist>=0:
			# print(dist)
			return True
		else:
			return False

	def intersection_union(self, cont1,  cont2):

		# returns union, intersection
		# contour stored in format top-left, top-right, bottom-right, bottom-left

		cont1 = cont1.reshape([cont1.shape[0], 2]).astype(np.float64)
		cont2 = cont2.reshape([cont2.shape[0], 2]).astype(np.float64)

		for i in range(4):
			if self.inside_point(cont1[i], cont2):
					return 1, 1
		for j in range(4):
			if self.inside_point(cont2[j], cont1):
				return 1, 1

		if cont1[0, 0] < cont2[0, 0] and cont1[1, 0] > cont2[1, 0] and cont2[0, 1] < cont1[0, 1] and cont2[2, 1] > cont1[2, 1]:
			return 1, 1

		if cont2[0, 0] < cont1[0, 0] and cont2[1, 0] > cont1[1, 0] and cont1[0, 1] < cont2[0, 1] and cont1[2, 1] > cont2[2, 1]:
			return 1, 1

		return 0, 0

	def get_list_of_files(self, dir_name):
		# create a list of file and sub directories 
		# names in the given directory 
		list_of_file = os.listdir(dir_name)
		all_files = list()
		# Iterate over all the entries
		for entry in list_of_file:
			# Create full path
			full_path = os.path.join(dir_name, entry)
			# If entry is a directory then get the list of files in this directory 
			if os.path.isdir(full_path):
				all_files = all_files + get_list_of_files(full_path)
			else:
				all_files.append(full_path)
					
		return all_files

	def watermark_text_on_image(self, image, text, pos,font):
		photo = image.copy()
		# make the image editable
		drawing = ImageDraw.Draw(photo)

		image_intensity = np.mean(np.array(image), axis=(0, 1)).astype(np.int32)

		
		r_value_choice=[]     
		if image_intensity[0]-64<0:
			r_value_choice=np.arange(image_intensity[0]+64,256)
		elif  image_intensity[0]+64>255:  
			r_value_choice=np.arange(0,image_intensity[0]-64)
		else :
			r_value_choice=np.append(np.arange(0,image_intensity[0]-64),np.arange(image_intensity[0]+64,255))
		
		g_value_choice=[]     
		if image_intensity[1]-64<0:
			g_value_choice=np.arange(image_intensity[1]+64,256)
		elif  image_intensity[1]+64>255:  
			g_value_choice=np.arange(0,image_intensity[1]-64)
		else :
			g_value_choice=np.append(np.arange(0,image_intensity[1]-64),np.arange(image_intensity[1]+64,255))
		
		b_value_choice=[]     
		if image_intensity[2]-64<0:
			b_value_choice=np.arange(image_intensity[2]+64,256)
		elif  image_intensity[2]+64>255:  
			b_value_choice=np.arange(0,image_intensity[2]-64)
		else :
			b_value_choice=np.append(np.arange(0,image_intensity[2]-64),np.arange(image_intensity[2]+64,255))
		
		r_value=np.random.choice(r_value_choice)
		g_value=np.random.choice(g_value_choice)
		b_value=np.random.choice(b_value_choice)
		drawing.text(pos, text, fill=(r_value,g_value,b_value), font=font)
   
		return photo

	def generate_watermark_on_images(self, output_image):  

		drawing = ImageDraw.Draw(output_image)		
		font_size_approx=(output_image.size[0]+output_image.size[1])//100 #taking approx size of font with respect to size of image

		prob = [1 for i in range(font_size_approx, font_size_approx*2)] + [0 for i in range(font_size_approx*2, font_size_approx*8)]+[1/10 for i in range(font_size_approx*8, 10*font_size_approx)]
		prob = np.array(prob).astype(np.float32)
		prob /= np.sum(prob)

		label=[]
		coordinate_label=[]

		margin=[output_image.size[0]//20,output_image.size[1]//20]
				
		x_of_watermark=margin[0] #initializing 
		y_of_watermark=margin[1] #initializing 

		while(y_of_watermark<output_image.size[1]-margin[1]):

			go_down=False
			while(go_down==False):

				spacing_y = output_image.size[1]//40+np.abs(np.random.normal(loc=0, scale=2*output_image.size[1]//40))
				spacing_x = output_image.size[0]//80+np.abs(np.random.normal(loc=0, scale=2*output_image.size[0]//80))

				could_not_enter = True
				always_overlapping = True

				for i in range(self.total_attempt):

					text,language=self.gen_word()

					# try:
					if language=="english":
						font_bytes = BytesIO(self.all_fonts[np.random.randint(len(self.all_fonts))])
					if language=="hindi":
						font_bytes = BytesIO(self.all_fonts_hindi[np.random.randint(len(self.all_fonts_hindi))])					
					
					font_size= np.random.choice(np.arange(font_size_approx,10*font_size_approx), p=prob)

					font = ImageFont.truetype(font_bytes,font_size)
					offset = font.getoffset(text)
					width, height = drawing.textsize(text, font=font)

					width -= offset[0]
					height -= offset[1]
					
					if x_of_watermark+width<output_image.size[0]-margin[0] and y_of_watermark+height<output_image.size[1]-margin[1]:  # to add condition that it goes to next line for x axis getting completed

						could_not_enter = False

						coordinates = np.zeros((4,1,2))

						coordinates[0][0][0]=x_of_watermark      #1st point
						coordinates[0][0][1]=y_of_watermark

						coordinates[1][0][0]=x_of_watermark+width  #2st point
						coordinates[1][0][1]=y_of_watermark

						coordinates[2][0][0]=x_of_watermark+width  #3rd point
						coordinates[2][0][1]=y_of_watermark+height

						coordinates[3][0][0]=x_of_watermark        #4th point
						coordinates[3][0][1]=y_of_watermark+height

						bbox = self.which_overlap(coordinates, coordinate_label)

						if bbox is None or len(coordinate_label)==0:
							always_overlapping = False
							output_image = self.watermark_text_on_image(output_image, text=text, pos=(x_of_watermark- offset[0], y_of_watermark - offset[1]),font=font)

							x_of_watermark += width+spacing_x

							if x_of_watermark >= output_image.size[0]-margin[0]: #as no word can be more accomodated so go_down=true
								go_down=True

							label.append(text)
							coordinate_label.append(coordinates)

							break

				if could_not_enter: 
					go_down=True

				if go_down==False and always_overlapping:

					x_of_watermark=spacing_x+bbox[1][0][0]

			x_of_watermark = margin[0]
			y_of_watermark += spacing_y

		output_final = np.array(output_image)
		return output_final,np.array(coordinate_label).astype(np.int32),label

	def create_annot1(self):

		all_paths = self.get_all_names_refresh()

		for i in all_paths:

			if os.path.exists(self.label_save_location+'.'.join(i.split('.')[:-1])+'.pkl'):
				continue
			print(i)
			image = Image.open(self.image_net_location+i).resize([768, 512]).convert('RGB')
			image = self.transparent(np.array(image),self.transparent_mean,self.transparent_gaussian)
			image = Image.fromarray(image.astype(np.uint8))
			final_image, coordinate_label ,label=self.generate_watermark_on_images(image)
			
			with open(self.label_save_location+'.'.join(i.split('.')[:-1])+'.pkl', 'wb') as f:
				pickle.dump([coordinate_label, label], f)
			plt.imsave(self.image_save_location+i, final_image)

	def split_ratio(self, all_list, train_ratio):

		#Forms the training and testing image paths, by splitting the dataset, deterministic due to self.seed()

		idx = np.arange(len(all_list))
		np.random.shuffle(idx)

		if os.path.exists(self.config['metadata']['ART']['meta']+'/train_files_'+str(train_ratio)+'.txt'):
			os.remove(self.config['metadata']['ART']['meta']+'/train_files_'+str(train_ratio)+'.txt')

		f = open(self.config['metadata']['ART']['meta']+'/train_files_'+str(train_ratio)+'.txt', 'w')

		train = np.array(all_list)[idx[:int(train_ratio*len(all_list))]]
		val = np.array(all_list)[idx[int(train_ratio*len(all_list)):]]
		for i in train:
			f.write(i+'\n')

		if os.path.exists(self.config['metadata']['ART']['meta']+'/test_files_'+str(train_ratio)+'.txt'):
			os.remove(self.config['metadata']['ART']['meta']+'/test_files_'+str(train_ratio)+'.txt')

		f = open(self.config['metadata']['ART']['meta']+'/test_files_'+str(train_ratio)+'.txt', 'w')

		for i in val:
			f.write(i+'\n')

	def create_annot(self):

		all_path = os.listdir('/home/Common/Mayank/Text/Segmentation/Dataset/ART/Images')

		all_list = ['.'.join(i.split('.')[:-1]) for i in all_path]
		self.split_ratio(all_list, float(self.split_type))
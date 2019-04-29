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

class ArtificialGen():

	def __init__(self, config, abc):
	# def __init__(self):

		self.config = config
		self.seed()

		self.all_words = words.words()
		self.english_alpha = ''.join([c for c in abc if c in 'abcdefghijklmnopqrstuvwxyz0123456789'])+"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		self.english_symbols = ''.join([c for c in abc if c not in 'abcdefghijklmnopqrstuvwxyz0123456789'])

		self.english = self.english_alpha + self.english_symbols

		self.transparent_mean = 0.8		
		self.transparent_gaussian = 0.06
		
		self.prob_lexi = 0.5
		self.symbol_word = 1

		self.art_font_size_range = self.config['augmentation']['font_range']
		self.border_range = self.config['augmentation']['border_range']

		self.font_dir_name='/home/Common/Datasets_SSD/Dataset_Text/ART/fonts_for_text'

		#probabilty distribution for length of words
		self.probability_dist = np.array([0.1, 0.6, 2.6, 5.2, 8.5, 12.2, 14, 14, 12.6, 10.1, 7.5])#, 5.2, 3.2, 2, 1, 0.6, 0.3, 0.2, 0.1, 0.1
		self.probability_dist = self.probability_dist/np.sum(self.probability_dist)

		list_of_files = self.get_list_of_files(self.font_dir_name)

		self.all_fonts = []
		for i in range(len(list_of_files)):
			with open(list_of_files[i],"rb") as f:
				font_bytes=f.read()
				self.all_fonts.append(font_bytes)

		self.image_net_location = "/media/mayank/0b40607e-7efc-4216-b12f-8bb86facfaed/Dataset_HDD/Image_Net/ILSVRC/Data/CLS-LOC/test/"
		# self.image_net_location = "/home/Common/ImageNet/test"
		self.images_orig = self.get_imagenet_images(self.config['augmentation']['imagenet_no'])#self.config['augmentation']['base_number']
		# self.image_save_location = '/home/Common/Mayank/Text/Segmentation/Dataset/ART/Images/'
		# self.label_save_location = '/home/Common/Mayank/Text/Segmentation/Dataset/ART/Labels/'

	def get_all_names_refresh(self):
	
		IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif']
		paths_of_image=[]
		
		for image in os.listdir(self.image_net_location):
			if(((image).split('.')[-1]).lower() in IMG_EXTENSIONS):
				paths_of_image.append(image)

		paths_of_image = np.array(paths_of_image)
		np.random.shuffle(paths_of_image)
		
		return paths_of_image.tolist()

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

	def get_imagenet_images(self, number):
		print("Loading ImageNet images....")
		l = np.arange(len(os.listdir(self.image_net_location)))
		np.random.shuffle(l)

		images = []
		idx = l[:number]

		for image in np.array(os.listdir(self.image_net_location))[idx]:
			images.append(np.array(Image.open(os.path.join(self.image_net_location, image)).convert('RGB')))
		print('Loaded ImageNet images')
		return images

	def transparent(self, x, mean, gaussian):

		alpha = np.random.normal(loc = mean, scale = gaussian, size=x.shape[0]*x.shape[1]*x.shape[2]).reshape([x.shape[0], x.shape[1], x.shape[2]])
		x = 255*alpha + x*(1 - alpha)
		return x

	def seed(self):

		np.random.seed(self.config['seed'])

	def generate_lexicon_dependent(self):

		while True:
			x = np.random.randint(len(self.all_words))
			text = self.all_words[x].lower()
			check = False
			for i in text:
				if i not in self.english:
					check = True
			if check:
				continue
			return text

	def generate_lexicon_free(self):

		length = np.random.choice(np.arange(1, 1+self.probability_dist.shape[0]), p=self.probability_dist)

		return self.generate_english(length)


	def generate_english(self, length):

		word = ''

		#special character in word
		if np.random.choice([0, 1], p=[self.symbol_word, 1 - self.symbol_word]) == 0:

			# symbol_length = [10, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
			# symbol_length = np.array(symbol_length).astype(np.float32)
			# symbol_length /= np.sum(symbol_length)
			symbol_final_no = np.random.choice(length+1)
			for i in range(length):
				word += self.english[np.random.randint(len(self.english))]

			# for i in range(symbol_final_no):
			# 	word += self.english_symbols[np.random.randint(len(self.english_symbols))]

			# while len(word) < length:
			# 	word += self.english_alpha[np.random.randint(len(self.english_alpha))]
		else:
			#no special character
			for i in range(length):
				word += self.english_alpha[np.random.randint(len(self.english_alpha))]

		return word.lower()

	def gen_word(self):

		if np.random.choice([0, 1], p=[self.prob_lexi, 1- self.prob_lexi]) == 0:
			return self.generate_lexicon_dependent()
		else:
			return self.generate_lexicon_free()

	def get_image(self):

		text = self.gen_word()

		font_size = np.random.randint(self.art_font_size_range[0], self.art_font_size_range[1])
		font_final = self.all_fonts[np.random.randint(len(self.all_fonts))]
		font_bytes = BytesIO(font_final)
		font = ImageFont.truetype(font_bytes, font_size)
		offset = font.getoffset(text)
		f_width, f_height = font.getsize(text)
		# f_width, f_height = font.textsize(text, font=font)

		f_width -= offset[0]
		f_height -= offset[1]
		img_width = f_width+np.random.randint(self.border_range[2], self.border_range[3]+1)
		img_height = f_height+np.random.randint(self.border_range[0], self.border_range[1]+1)
		tries = 0
		
		while 1:
			img = self.images_orig[np.random.randint(len(self.images_orig))]
			
			if tries > 10:
				return None, None

			if img.shape[0] - 1 < img_height or img.shape[1] -1 < img_width:
				tries += 1
				continue

			start_x = np.random.randint(img.shape[0]-img_height)
			start_y = np.random.randint(img.shape[1]-img_width)
			break

		crop = img[start_x:start_x+img_height, start_y:start_y+img_width, :]
		crop = cv2.filter2D(crop,-1,np.ones([5, 5], dtype=np.float32)/25)
		crop_img = Image.fromarray(crop, 'RGB')
		
		drawing = ImageDraw.Draw(crop_img)

		pos = ((img_width-f_width)//2 - offset[0], (img_height-f_height)//2 - offset[1])
		###Text color
		image_intensity = np.mean(np.array(crop_img), axis=(0, 1)).astype(np.int32)

		r_value_choice=[]     
		if image_intensity[0]-96<0:
			r_value_choice=np.arange(image_intensity[0]+96,256)
		elif  image_intensity[0]+96>255:  
			r_value_choice=np.arange(0,image_intensity[0]-96)
		else :
			r_value_choice=np.append(np.arange(0,image_intensity[0]-96),np.arange(image_intensity[0]+96,255))
		
		g_value_choice=[]     
		if image_intensity[1]-96<0:
			g_value_choice=np.arange(image_intensity[1]+96,256)
		elif  image_intensity[1]+96>255:  
			g_value_choice=np.arange(0,image_intensity[1]-96)
		else :
			g_value_choice=np.append(np.arange(0,image_intensity[1]-96),np.arange(image_intensity[1]+96,255))
		
		b_value_choice=[]     
		if image_intensity[2]-96<0:
			b_value_choice=np.arange(image_intensity[2]+96,256)
		elif  image_intensity[2]+96>255:  
			b_value_choice=np.arange(0,image_intensity[2]-96)
		else :
			b_value_choice=np.append(np.arange(0,image_intensity[2]-96),np.arange(image_intensity[2]+96,255))
		
		r_value=np.random.choice(r_value_choice)
		g_value=np.random.choice(g_value_choice)
		b_value=np.random.choice(b_value_choice)

		drawing.text(pos, text, fill=(r_value, g_value, b_value), font=font)
		width, height = drawing.textsize(text, font=font)

		return crop_img, text
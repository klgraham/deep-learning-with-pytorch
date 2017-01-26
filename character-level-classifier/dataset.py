from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split

import torch
import glob
import unicodedata
import string

alphabet = string.ascii_letters + " .,;'"
alphabet_size = len(alphabet)

# Use scikit-learn to facilitate loading the data

def find_files(path):
	return glob.glob(path)

def unicode_to_ascii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)

# build the category to line dictionary

def read_lines(fileline):
	lines = open(fileline).read().strip().split('\n')
	return [unicode_to_ascii(line) for line in lines]

def load_data(path_to_data):
	category_to_lines = {}
	categories = []
	
	for fileline in find_files(path_to_data):
		category = fileline.split('/')[-1].split('.')[0]
		categories.append(category)

		lines = read_lines(fileline)
		category_to_lines[category] = lines

	return categories, category_to_lines   
		
#def bundle_data(data, categories):
#	category_to_lines = {}
#	
#	for n in range(len(categories)):
#		category = categories[n]
#		category_to_lines[n]
#		lines = data[n].strip().split('\n')
#		
#		for line in lines:
#			ascii_line = unicode_to_ascii(line)
			
	
# Load dataset using scikit-learn, with a 70/30 train/test	split
# this is a smarter combo of load_data
def load(data_dir):
	dataset = load_files('data_dir', load_content=True, encoding='UTF-8', decode_error='replace')

	X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=1354925401)
	
	# dataset.target_names is a list of the category names
	# dataset.target is a list of indices into the above list
	# X and y are lists of the datafile's contents and the index of the category, respectively
#	
#	category_to_lines_train = {}
#	categories_train = []
#	category_to_lines_test = {}
#	categories_test = []
#	
#	for n in range(len(t_train)):
		
	
	return dataset.target_names, X_train, X_test, y_train, y_test

# get index of each letter in the alphabet
def letter_to_index(letter):
	return alphabet.find(letter)

# for example, get the one-hot encoding for a letter
def letter_to_tensor(letter):
	t = torch.zeros(1, alphabet_size)
	t[0][letter_to_index(letter)] = 1
	return t

def line_to_tensor(line):
	# one row for each letter, a single column, then the one-hot elements extend into the plane
	line_tensor = torch.zeros(len(line), 1, alphabet_size)
	for character_index, character in enumerate(line):
		line_tensor[character_index][0][letter_to_index(character)] = 1
	return line_tensor
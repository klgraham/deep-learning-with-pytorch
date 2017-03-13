from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch
import string

# these are the supported characters
alphabet = string.ascii_letters + string.punctuation + string.digits
alphabet_size = len(alphabet)

# get index of each letter in the alphabet
def char2index(letter):
    return alphabet.find(letter)

# Load dataset using scikit-learn, with a 70/30 train/test	split
# this is a smarter combo of load_data
# may need to load with 'ISO-8859-1'  instead of UTF-8
def load(data_dir, encoding='us-ascii'):
	dataset = load_files(data_dir, load_content=True, encoding=encoding, decode_error='replace')

    # dataset.target_names is a list of the category names
	# dataset.target is a list of indices into the above list
	# X and y are lists of the datafile's contents and the index of the category, respectively
	X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=1354925401)
	
	return dataset.target_names, X_train, X_test, y_train, y_test

# build the category to line dictionary

# read all lines in the file, return an array of lines
def get_lines(data_as_string):
    lines = data_as_string.strip().split('\n')
    return [line for line in lines]


# convert each line of a file into a one-hot vector
# a line can be a minibatch

def line2tensor(line):
    line_tensor = torch.zeros(len(line), 1, alphabet_size)
    for index_in_line, character in enumerate(line):
        index_in_vocab = char2index(character)
        if index_in_vocab >= 0:
            line_tensor[index_in_line][0][index_in_vocab] = 1
    return line_tensor

# one-hot encode the labels
# this is not needed here
#def label2vector(label_assignments):
#    onehot_encoder = preprocessing.LabelBinarizer()
#    onehot_encoder.fit(label_assignments)
#    return onehot_encoder

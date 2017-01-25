import torch
import glob
import unicodedata
import string

alphabet = string.ascii_letters + " .,;'"
alphabet_size = len(alphabet)

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
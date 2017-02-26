import torch
#import pickle
from data import *
from model import *
from random import randint
import time
import math
import sys
import numpy as np
import os

def output2label(output, labels):
    top_scores, top_index = output.data.topk(1)
    index_of_best_category = top_index[0][0]
    return labels[index_of_best_category], index_of_best_category   

# select a random feature/label pair
# a single feature is made up of several lines, so need to pick one
def get_random_index(vector):
    return randint(0, len(vector) - 1)

def random_training_pair(training_features, training_labels):
    r = get_random_index(training_features)
    
    features = training_features[r]
    
    # features is made of several lines, so pick one of them
    lines = get_lines(features)
    random_line = lines[randint(0, len(lines) - 1)]
    
    label = training_labels[r]   
    
    line_tensor = Variable(line2tensor(random_line))
    label_tensor = Variable(torch.LongTensor([np.asscalar(label)]))
    
    return random_line, label, line_tensor, label_tensor 

# loss function here is negative log-liklihood
criterion = nn.NLLLoss() 

learning_rate = 0.005

# this treats each file as a mini-batch
def train_on_file(rnn, label_tensor, line_tensor):
    # line_tensor has dimension (num_letters in line) x 1 x alphabet_size
    
    # initialize the mini-batch
    hidden = rnn.initHidden()
    rnn.zero_grad()
    
    # loop over each character in the line
    num_characters = line_tensor.size()[0]
    for i in range(num_characters):
        output, hidden = rnn(line_tensor[i], hidden)
        
    # compute loss and backpropagate
    loss = criterion(output, label_tensor)
    loss.backward()
    
    # update parameters' gradients
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
        
    return output, loss.data[0]
    
def train(data_dir, model_name, encoding='us-ascii'):
    base_dir = '../data/train_test/' + model_name
    
    # check to see if data has been loaded already
    try:
        X_train = torch.load(base_dir + '/features.train.dat')
        X_test = torch.load(base_dir + '/features.test.dat')
        y_train = torch.load(base_dir + '/labels.train.dat')
        y_test = torch.load(base_dir + '/labels.test.dat')
        labels = torch.load(base_dir + '/all-labels.dat')
        print('Loaded cached data')
        
    except IOError as e:
        print('Reading data from disk')
        labels, X_train, X_test, y_train, y_test = load(data_dir, encoding=encoding)

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        torch.save(X_train, base_dir + '/features.train.dat')
        torch.save(X_test, base_dir + '/features.test.dat')
        torch.save(y_train, base_dir + '/labels.train.dat')
        torch.save(y_test, base_dir + '/labels.test.dat')
        torch.save(labels, base_dir + '/all-labels.dat')
    
    num_labels = len(labels)
    
    num_hidden = 128
    num_epochs = 100000
    print_every = 1000
    plot_every = 1000
    
    rnn = CharRNN(alphabet_size, num_hidden, num_labels)

    # track loss for plotting
    current_loss = 0
    losses = []

    def time_since(t):
        now = time.time()
        seconds = now - t
        minutes = math.floor(seconds / 60)
        seconds -= minutes * 60
        return '%dm %ds' % (minutes, seconds)

    start = time.time()

    for epoch in range(1, num_epochs + 1):
        line, label, line_tensor, label_tensor = random_training_pair(X_train, y_train)

        if len(line_tensor.size()) > 0:
            output, loss = train_on_file(rnn, label_tensor, line_tensor)
            current_loss += loss

            # print progress
            if epoch % print_every == 0:
                prediction, prediction_index = output2label(output, labels)
                correct = 'YES' if prediction_index == label else 'NO (%s)' % label
                print('epoch: %d (%d%%, %s) %.4f %s / %s' % (epoch, epoch / num_epochs * 100, time_since(start), loss, prediction, correct))

            # store average losses
            if epoch % plot_every == 0:
                losses.append(current_loss / plot_every)
                current_loss = 0

    torch.save(rnn, model_name + '-char-rnn-classifier.model')
    
    # evaluate on test set
    num_correct = 0
    
    for i in range(len(y_test)):
        label = y_test[i]
        features = X_test[i]
        
        lines = get_lines(features)
        output = None
        hidden = rnn.initHidden()
        for line in lines:
            line_tensor = Variable(line2tensor(line))
            
            if len(line_tensor.size()) > 0:
                for i in range(line_tensor.size()[0]):
                    output, hidden = rnn(line_tensor[i], hidden)
           
        prediction, prediction_index = output2label(output, labels)
        
        if label == prediction_index:
            num_correct += 1
            
    print('Test accuracy: %.4f' % (num_correct/len(y_test) * 100))
        
        
    
    
if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2])
import torch
from data import *
from model import *
from random import randint
import time
import math
import sys

def output_to_category(output, categories):
    top_scores, top_index = output.data.topk(1)
    index_of_best_category = top_index[0][0]
    return categories[index_of_best_category], index_of_best_category
    
def get_random_element(vector):
    return vector[randint(0, len(vector) - 1)]

def random_training_pair(categories, category_to_lines):
    category = get_random_element(categories)
    line = get_random_element(category_to_lines[category])
    category_tensor = Variable(torch.LongTensor([categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor

# loss function here is negative log-liklihood
criterion = nn.NLLLoss() 

learning_rate = 0.005

# this treats each line as a mini-batch
def train_on_line(rnn, category_tensor, line_tensor):
    # line_tensor has dimension (num_letters in line) x 1 x alphabet_size
    
    # initialize the mini-batch
    hidden = rnn.initHidden()
    rnn.zero_grad()
    
    # loop over each letter in the line
    num_letters = line_tensor.size()[0]
    for i in range(num_letters):
        output, hidden = rnn(line_tensor[i], hidden)
        
    # compute loss and backpropagate
    loss = criterion(output, category_tensor)
    loss.backward()
    
    # update parameters' gradients
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
        
    return output, loss.data[0]
    
def train(data_dir):
    data_glob = data_dir + "*.txt"
    categories, category_to_lines = load_data(data_glob)
    torch.save(categories, 'categories.dat')
    torch.save(category_to_lines, 'category_to_lines.dat')
    
    num_categories = len(categories)
    
    num_hidden = 128
    num_epochs = 100000
    print_every = 5000
    plot_every = 1000
    
    rnn = CharRNN(alphabet_size, num_hidden, num_categories)

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
        category, line, category_tensor, line_tensor = random_training_pair(categories, category_to_lines)
        output, loss = train_on_line(rnn, category_tensor, line_tensor)
        current_loss += loss

        # print progress
        if epoch % print_every == 0:
            prediction, prediction_index = output_to_category(output, categories)
            correct = 'YES' if prediction == category else 'NO (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / num_epochs * 100, time_since(start), loss, line, prediction, correct))

        # store average losses
        if epoch % plot_every == 0:
            losses.append(current_loss / plot_every)
            current_loss = 0

    torch.save(rnn, 'char-rnn-classifier.model')
    
    
if __name__ == '__main__':
    train(sys.argv[1])
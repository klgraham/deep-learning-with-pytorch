from model import *
from data import *
import sys
import math

rnn = torch.load('char-rnn-classifier.model')
categories = torch.load('categories.dat')
category_to_lines = torch.load('category_to_lines.dat')

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        
    return output

def predict(line, num_predictions=3):
    print('\n', line)
    output = evaluate(Variable(line_to_tensor(line)))
    
    # get top n predictions
    top_scores, top_category_indices = output.data.topk(num_predictions, 1)
    
    predictions = []
    for i in range(num_predictions):
        score = math.exp(top_scores[0][i])
        category_index = top_category_indices[0][i]      
        print('\t(%.2f) %s' % (score, categories[category_index]))            
        predictions.append([score, categories[category_index]])
        
    return predictions

if __name__ == '__main__':
    predict(sys.argv[1], 5)
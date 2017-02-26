from model import *
from data import *
import sys
import math

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        
    return output

def predict(model_name, line, num_predictions=3):

    # load model
    rnn = torch.load(model_name + '-char-rnn-classifier.model')
    labels = torch.load(model_name + '-labels.dat')

    print('\n', line)
    output = evaluate(Variable(line_to_tensor(line)))
    
    # get top n predictions
    top_scores, top_label_indices = output.data.topk(num_predictions, 1)
    
    predictions = []
    for i in range(num_predictions):
        score = math.exp(top_scores[0][i])
        label_index = top_label_indices[0][i]      
        print('\t(%.2f) %s' % (score, labels[label_index]))            
        predictions.append([score, labels[label_index]])
        
    return predictions

if __name__ == '__main__':
    predict(sys.argv[1], sys.argv[2])
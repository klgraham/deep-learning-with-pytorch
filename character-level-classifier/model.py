import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.hidden_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.hidden_layer(combined)
        output = self.output_layer(combined)
        output = self.softmax(output)
        return output, hidden
    
    # used to initialize previous hidden state
    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
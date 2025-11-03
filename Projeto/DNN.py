import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        self.activations = nn.ModuleList()

        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_size, num_classes))
        else:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.activations.append(nn.ReLU())        
            
            for i in range(len(hidden_sizes)-1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                self.activations.append(nn.ReLU())

            self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
    
    def forward(self, x, use_nll=False):
        x = x.view(x.size(0), -1)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activations[i](x)
            x = self.dropout(x)

        x = self.layers[-1](x)
        if use_nll:
            x = F.log_softmax(x, dim=1)
        return x
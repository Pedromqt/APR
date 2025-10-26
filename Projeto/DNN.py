import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
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
    
    def forward(self, x):
        out = x 
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i < len(self.activations):
                out = self.activations[i](out)
        return out
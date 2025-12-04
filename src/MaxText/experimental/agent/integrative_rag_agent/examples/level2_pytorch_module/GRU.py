import torch
import torch.nn as nn

class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

 # GRU layer, which is often faster to train than LSTM
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
 
 # Readout layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
 # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
 
 # Forward propagate GRU
        out, _ = self.gru(x, h0)
 
 # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
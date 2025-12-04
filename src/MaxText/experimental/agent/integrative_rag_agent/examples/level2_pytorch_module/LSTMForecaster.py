import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
 
 # LSTM layer: batch_first=True makes the input/output tensors have shape (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
 
 # Fully connected layer to map the LSTM output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
 # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
 
 # Pass input through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))
 
 # We take the output from the last time step for prediction
        out = self.fc(out[:, -1, :])
        return out
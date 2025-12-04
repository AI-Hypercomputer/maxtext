import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_len, hidden_size)
        energy = torch.tanh(self.attn(lstm_outputs))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(lstm_outputs.size(0), 1).unsqueeze(1)
        attn_weights = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attn_weights, dim=1)

class LSTMAttentionForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMAttentionForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        output = self.fc(context)
        return output
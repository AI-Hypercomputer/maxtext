import torch
import torch.nn as nn

# This example uses the built-in PyTorch module for simplicity.
# It encapsulates Multi-Head Attention and a Feed-Forward Network.
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Embedding(ntoken, ninp)
        encoder_layers = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

    def forward(self, src, src_mask=None):
        src = self.encoder(src) * torch.sqrt(torch.tensor(self.ninp))
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
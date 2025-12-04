import torch.nn as nn
import torch.nn.functional as F

class CreditScoringMLP(nn.Module):
    def __init__(self, input_features):
        super(CreditScoringMLP, self).__init__()
        self.layer1 = nn.Linear(input_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
 
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
 
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
 
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
 
 # Output a single logit, which can be passed to a sigmoid for a probability
        x = self.layer3(x)
        return x
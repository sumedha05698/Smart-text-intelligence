import torch
import torch.nn as nn

class ANNClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, 10)
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

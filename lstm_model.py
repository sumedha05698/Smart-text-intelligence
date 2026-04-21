import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Embedding layer: word id -> word meaning
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 1,
            embedding_dim=10,
            padding_idx=0
        )

        # LSTM layer: reads words in order
        self.lstm = nn.LSTM(
            input_size=10,
            hidden_size=16,
            batch_first=True
        )

        # Final classification layer
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        # x: (batch_size, sequence_length)
        x = self.embedding(x)
        # x: (batch_size, sequence_length, embedding_dim)

        _, (h, _) = self.lstm(x)
        # h: (1, batch_size, hidden_size)

        out = self.fc(h[-1])
        return torch.sigmoid(out)


import torch
import torch.nn as nn

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size + 1,
            10,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=10,
            hidden_size=16,
            batch_first=True
        )

        self.attention = nn.Linear(16, 1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x, return_attention=False):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)

        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)

        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        out = torch.sigmoid(out)

        if return_attention:
            return out, attn_weights

        return out

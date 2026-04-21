import torch
import torch.nn as nn
from ann_model import ANNClassifier
from utils import tokenize, build_vocab, encode,pad_sequences

# Load data
texts = []
labels = []

with open("data/reviews.txt") as f:
    for line in f:
        text, label = line.strip().split("|")
        texts.append(tokenize(text))
        labels.append(int(label))

vocab = build_vocab(texts)

X = [encode(t, vocab) for t in texts]
print("ENCODED X (before padding):", X)
X = pad_sequences(X)
print("ENCODED X (after padding):", X)
X = torch.tensor(X)
y = torch.tensor(labels).float().unsqueeze(1)

model = ANNClassifier(len(vocab))
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    preds = model(X)
    loss = loss_fn(preds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training complete")

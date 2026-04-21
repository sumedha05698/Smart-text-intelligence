import torch
import torch.nn as nn

from utils import tokenize, build_vocab, encode, pad_sequences
from lstm_model import LSTMAttentionClassifier

# Load data
texts = []
labels = []

with open("data/reviews.txt") as f:
    for line in f:
        text, label = line.strip().split("|")
        texts.append(tokenize(text.strip()))
        labels.append(int(label.strip()))

# Build vocab
vocab = build_vocab(texts)

# Encode + pad
X = [encode(t, vocab) for t in texts]
X = pad_sequences(X)
X = torch.tensor(X)

y = torch.tensor(labels).float().unsqueeze(1)

# Model
model = LSTMAttentionClassifier(len(vocab))
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(300):
    preds = model(X)
    loss = loss_fn(preds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Attention Epoch {epoch}, Loss: {loss.item():.4f}")

print("Attention-based LSTM training complete")

def show_attention(sentence, model, vocab):
    model.eval()

    tokens = tokenize(sentence)
    encoded = encode(tokens, vocab)
    padded = pad_sequences([encoded])
    tensor = torch.tensor(padded)

    with torch.no_grad():
        prediction, attn_weights = model(tensor, return_attention=True)

    attn_weights = attn_weights.squeeze().numpy()

    print(f"\nSentence: {sentence}")
    print("Attention weights:")
    for word, weight in zip(tokens, attn_weights):
        print(f"{word:10s} → {weight:.3f}")

show_attention("This movie is not good", model, vocab)
show_attention("Amazing acting and story", model, vocab)
show_attention("Worst movie ever", model, vocab)

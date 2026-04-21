import torch
import torch.nn as nn

from utils import tokenize, build_vocab, encode, pad_sequences
from lstm_model import LSTMClassifier

# -----------------------------
# 1️⃣ Load data
# -----------------------------
texts = []
labels = []

with open("data/reviews.txt") as f:
    for line in f:
        text, label = line.strip().split("|")
        texts.append(tokenize(text.strip()))
        labels.append(int(label.strip()))

# -----------------------------
# 2️⃣ Build vocabulary
# -----------------------------
vocab = build_vocab(texts)

# -----------------------------
# 3️⃣ Encode + Pad
# -----------------------------
X = [encode(t, vocab) for t in texts]
X = pad_sequences(X)
X = torch.tensor(X)

y = torch.tensor(labels).float().unsqueeze(1)

# -----------------------------
# 4️⃣ Model, loss, optimizer
# -----------------------------
model = LSTMClassifier(len(vocab))
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 5️⃣ Training loop
# -----------------------------
for epoch in range(300):
    preds = model(X)
    loss = loss_fn(preds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"LSTM Epoch {epoch}, Loss: {loss.item():.4f}")

print("LSTM training complete")



# -------------------------
# 6️⃣ Make real predictions
# -------------------------

def predict_sentence(sentence, model, vocab):
    model.eval()  # evaluation mode

    tokens = tokenize(sentence)
    encoded = encode(tokens, vocab)
    padded = pad_sequences([encoded])
    tensor = torch.tensor(padded)

    with torch.no_grad():
        prediction = model(tensor)

    return prediction.item()


test_sentences = [
    "I love this movie",
    "This movie is not good",
    "Worst film ever",
    "Amazing experience",
    "Not worth watching"
]

print("\n--- Predictions ---")
for s in test_sentences:
    score = predict_sentence(s, model, vocab)
    label = "Positive 🙂" if score > 0.5 else "Negative 🙁"
    print(f"{s} → {label} (score={score:.3f})")

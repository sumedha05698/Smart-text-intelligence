def tokenize(text):
    return text.lower().split()


def build_vocab(sentences):
    vocab = {}
    idx = 1  # 0 reserved for padding
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def encode(sentence, vocab):
    return [vocab[word] for word in sentence if word in vocab]


def pad_sequences(sequences):
    max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        seq = seq + [0] * (max_len - len(seq))
        padded.append(seq)

    return padded

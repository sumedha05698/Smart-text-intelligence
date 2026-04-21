from transformers import pipeline

# Load pretrained BERT sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Test sentences
sentences = [
    "I love this movie",
    "This movie is not good",
    "Worst film ever",
    "Amazing acting and story",
    "Not worth watching"
]

# Run predictions
for s in sentences:
    result = sentiment_model(s)[0]
    print(f"{s}")
    print(f"Prediction: {result['label']} (confidence={result['score']:.3f})\n")

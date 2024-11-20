import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "sentiment_analysis/sentiment_model/fine_tuned_financial_bert_combined"
TOKENIZER_DIR = "sentiment_analysis/sentiment_model/fine_tuned_financial_bert_combined"

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

def analyze_sentiment(texts):
    """
    Analyze sentiment using the fine-tuned BERT model.
    
    Args:
        texts (list[str]): List of text inputs to analyze.
    
    Returns:
        list[str]: List of sentiment predictions (Bullish, Bearish, Neutral).
    """
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    sentiment_labels = ["Bearish", "Bullish", "Neutral"]
    return [sentiment_labels[pred] for pred in predictions]

if __name__ == "__main__":
    example_texts = [
        "Stock prices are expected to rise with new product releases.",
        "The company's earnings report indicates major losses.",
        "Market conditions remain stable."
    ]
    sentiments = analyze_sentiment(example_texts)
    print(sentiments)

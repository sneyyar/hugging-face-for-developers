from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("yelp_polarity")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
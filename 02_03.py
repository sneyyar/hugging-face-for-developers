from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset, load_metric

dataset = load_dataset("yelp_polarity")
test_dataset = dataset["test"].select(range(100))

model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_yelp_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_yelp_model")

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_test = test_dataset.map(preprocess_function, batched=True)


metric = load_metric("accuracy", trust_remote_code=True)

def compute_metrics(p):
    logits, labels = p
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate()
print("Results", results)
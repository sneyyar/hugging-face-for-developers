from datasets import load_dataset

dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

for i, batch in enumerate(dataset):
    print(batch)
    if i > 3:
        break
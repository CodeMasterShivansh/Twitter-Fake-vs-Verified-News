import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# -----------------------------
# Load Cleaned Dataset
# -----------------------------
print("Loading cleaned_data.csv...")
df = pd.read_csv("data/cleaned_data.csv")

texts = df["clean_title"].tolist()
labels = df["label"].tolist()

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# -----------------------------
# Tokenization
# -----------------------------
print("Tokenizing...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=64)

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

# -----------------------------
# Torch Dataset Class
# -----------------------------
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(train_encodings, train_labels)
test_dataset = FakeNewsDataset(test_encodings, test_labels)

# -----------------------------
# Metrics Function
# -----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# -----------------------------
# Load BERT Model
# -----------------------------
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# -----------------------------
# Training Settings
# -----------------------------
training_args = TrainingArguments(
    output_dir="models/bert_finetuned",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# -----------------------------
# Train and Evaluate
# -----------------------------
print("Training started...")
trainer.train()

print("Evaluating model...")
results = trainer.evaluate()
print("Evaluation Results:", results)

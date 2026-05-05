from transformers import AutoTokenizer
import transformers
print("Transformers version:", transformers.__version__)
from transformers import TrainingArguments
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np
from pathlib import Path

# ==== 1. Dummy Data ====
data = {
    "text": [
        "Oh great, another Monday. Just what I needed!",
        "I love getting stuck in traffic every day.",
        "What a beautiful day!",
        "I'm so excited to do my taxes.",
        "Wow, your idea is just brilliant. Not.",
        "I'm genuinely happy today.",
        "Nothing says fun like cleaning the house.",
        "This is exactly what I hoped wouldn't happen.",
        "Thank you so much for breaking my phone!",
        "I'm looking forward to more work this weekend."
    ],
    "label": [1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
}
dataset = Dataset.from_dict(data).train_test_split(test_size=0.2)

# ==== 2. Tokenizer ====
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=16)

tokenized = dataset.map(tokenize, batched=True)

# ==== 3. Model ====
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# ==== 4. Metrics ====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prf = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": prf[0], "recall": prf[1], "f1": prf[2]}

# ==== 5. Training ====
args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="epoch",  👈 REMOVE or comment this
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    logging_steps=5,
    # save_strategy="epoch", 👈 Optional, also remove if it causes error
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ==== 6. Save Model ====
output_dir = "./sarcasm_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ Model and tokenizer saved.")

# ==== 7. Export to ONNX ====
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

onnx_path = Path("sarcasm_model.onnx")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer.save_pretrained("./tokenizer")

# Define model config
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature="sequence-classification")
onnx_config = model_onnx_config(model.config)

export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=14,
    output=onnx_path
)

print(f"✅ Exported ONNX model to: {onnx_path}")


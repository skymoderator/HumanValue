import os
from typing import Dict, List, Optional
from torch.utils.data import Dataset

os.environ["HF_HOME"] = os.path.join(os.getcwd())
# os.environ["WANDB_PROJECT"] = "MSBD5018 Group Project"
# os.environ["WANDB_LOG_MODEL"] = "all"
import transformers
from datasets import load_dataset
import time
import numpy as np
import json
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

labels = [
    k
    for k in load_dataset(
        "webis/Touche23-ValueEval", name="value-categories", split="meta"
    )[0].keys()
]
"""
labels = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: caring",
    "Benevolence: dependability",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
    "Universalism: objectivity",
]
"""
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
dataset = load_dataset(path="webis/Touche23-ValueEval")
"""DatasetDict({
    train: Dataset({
        features: ['Argument ID', 'Conclusion', 'Stance', 'Premise', 'Labels'],
        num_rows: 5393
    })
    validation: Dataset({
        features: ['Argument ID', 'Conclusion', 'Stance', 'Premise', 'Labels'],
        num_rows: 1896
    })
    test: Dataset({
        features: ['Argument ID', 'Conclusion', 'Stance', 'Premise', 'Labels'],
        num_rows: 1576
    })
})
"""
model_path = "roberta-large"
# model_path = "results/mlm"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.RobertaForMaskedLM.from_pretrained(model_path)
datacollator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer)
block_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_texts(examples):
    batch_encoding = tokenizer(
        text=examples["Premise"],
        text_pair=examples["Conclusion"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    batch_encoding.to(device)
    return batch_encoding


dataset = dataset.map(
    process_texts, batched=True, remove_columns=dataset["train"].column_names
)


def compute_metrics(eval_pred):
    """accuracy for mlm"""
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
    }


def inference_model(model):
    model.eval()
    text = tokenizer.decode(
        dataset["test"][0]["input_ids"], skip_special_tokens=True
    ).split(" ")
    text[np.random.randint(0, len(text))] = tokenizer.mask_token
    text = " ".join(text)
    input = tokenizer(text, return_tensors="pt")
    input.to(device)
    mask_token_index = torch.where(input["input_ids"] == tokenizer.mask_token_id)[1]
    with torch.no_grad():
        logits = model(**input).logits
    mask_token_logits = logits[0, mask_token_index, :]
    top3 = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()
    print(text)
    for i, token in enumerate(top3):
        print(
            i,
            text.replace(tokenizer.mask_token, tokenizer.decode([token])),
        )


if __name__ == "__main__":
    fake_batch_size = 256
    real_batch_size = 2
    num_epochs = 20
    training_args = transformers.TrainingArguments(
        # output_dir=f"./results/{time.strftime('%Y%m%d_%H%M%S')}",
        output_dir="./results/mlm-large",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=real_batch_size,
        per_device_eval_batch_size=real_batch_size,
        gradient_accumulation_steps=fake_batch_size // real_batch_size,
        weight_decay=0.001,
        logging_dir="./logs",
        fp16=True,
        logging_strategy="steps",
        logging_steps=1,
        max_grad_norm=1.0,
        push_to_hub=False,
        report_to="none",
        run_name="mlm-large",
        do_train=True,
        save_strategy="epoch",
        save_total_limit=2,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=datacollator,
        # compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()
    inference_model(trainer.model)

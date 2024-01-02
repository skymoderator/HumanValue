import os
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset

os.environ["HF_HOME"] = os.path.join(os.getcwd())
os.environ["WANDB_PROJECT"] = "MSBD5018 Group Project"
os.environ["WANDB_LOG_MODEL"] = "all"
import transformers
from datasets import load_dataset
import time
import numpy as np
import json
import torch
from transformers import Trainer, DataCollatorForLanguageModeling
from util import *
from transformers.models.roberta.modeling_roberta import MaskedLMOutput
from functools import partial
from sklearn.utils import class_weight, compute_class_weight
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deberta.modeling_deberta import (
    DebertaPredictionHeadTransform,
    DebertaForMaskedLM,
)
from torch import nn
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

"""
labels = [
    k
    for k in load_dataset(
        "webis/Touche23-ValueEval", name="value-categories", split="meta"
    )[0].keys()
]
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
labels = [
    "thought",
    "action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "dominance",
    "resources",
    "Face",
    "personal",
    "societal",
    "Tradition",
    "rules",
    "interpersonal",
    "Humility",
    "caring",
    "dependability",
    "concern",
    "nature",
    "tolerance",
    "objectivity",
]
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


dataset = dataset.filter(lambda e: np.any(np.array(e["Labels"])))
dataset_labels = torch.tensor(dataset["train"]["Labels"], dtype=torch.float64)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(id2label)),
    y=np.array(dataset["train"]["Labels"]).nonzero()[1],
)

model_path = "microsoft/deberta-large-mnli"


def data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = tokenizer.pad(
        features,
        padding="longest",
        # max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    return batch


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path, id2label=id2label, label2id=label2id
)
config = transformers.AutoConfig.from_pretrained(
    model_path, num_labels=len(labels), id2label=id2label, label2id=label2id
)
# config.tie_word_embeddings = False
model = DebertaForMaskedLM.from_pretrained(model_path, config=config)
# model = RobertaForMaskedPromptBasedLM.from_pretrained(
#     model_path, local_files_only=True, config=config
# )
# model.loss_fct = torch.nn.BCEWithLogitsLoss()
# model.lm_head.decoder = torch.nn.Linear(config.hidden_size, config.num_labels)
# model.lm_head.bias = torch.nn.Parameter(torch.zeros(config.num_labels))
# model.lm_head.decoder.bias = model.lm_head.bias
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_data_func(examples):
    batch_encoding = tokenizer(
        text=[
            "Premise is that "
            + premise
            + " Conclusion is that "
            + conclusion
            + ". So, is the context related to the "
            + tokenizer.mask_token
            + "?"
            for premise, conclusion in zip(examples["Premise"], examples["Conclusion"])
        ],
        # text=[
        #     "Premise: "
        #     + premise
        #     + " Conclusion: "
        #     + conclusion
        #     + ". Topic: "
        #     + tokenizer.mask_token
        #     for premise, conclusion in zip(examples["Premise"], examples["Conclusion"])
        # ],
        truncation=True,
        # padding="max_length",
        padding="do_not_pad",
        # max_length=tokenizer.model_max_length,
        # return_tensors="pt",
    )
    # batch_encoding["labels"] = torch.tensor(examples["Labels"], dtype=torch.float32)
    # batch_encoding["masked_position"] = torch.where(
    #     batch_encoding["input_ids"] == tokenizer.mask_token_id
    # )[1]
    # batch_encoding["masked_position"] = torch.tensor(
    #     [
    #         [
    #             i
    #             for i, token_id in enumerate(ids)
    #             if token_id == tokenizer.mask_token_id
    #         ][0]
    #         for ids in batch_encoding["input_ids"]
    #     ]
    # )
    # batch_encoding.to(device)
    return batch_encoding


dataset = dataset.map(
    process_data_func,
    batched=True,
)


if __name__ == "__main__":
    fake_batch_size = 256
    real_batch_size = 2
    num_epochs = 30
    eval_steps = len(dataset["train"]) // fake_batch_size
    training_args = transformers.TrainingArguments(
        # output_dir=f"./results/{time.strftime('%Y%m%d_%H%M%S')}",
        output_dir="./results/deberta-large_mlm",
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
        do_train=True,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        # report_to="none",
        run_name="deberta-large_mlm",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        # metric_for_best_model="micro_f1",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        # compute_metrics=partial(compute_metrics, id2label),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=0.15
        ),
    )
    trainer.train()
    # save_prediction(trainer=trainer, dataset=dataset, split="test", id2label=id2label)
    # save_prediction(
    #     trainer=trainer, dataset=dataset, split="validation", id2label=id2label
    # )
    # save_prediction(trainer=trainer, dataset=dataset, split="train", id2label=id2label)

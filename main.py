import os
from typing import Dict, List, Optional
from torch.utils.data import Dataset

os.environ["HF_HOME"] = os.path.join(os.getcwd())
os.environ["WANDB_PROJECT"] = "MSBD5018 Group Project"
os.environ["WANDB_LOG_MODEL"] = "all"
import transformers
from datasets import load_dataset
import time
import numpy as np
import json
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from util import *
from functools import partial

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


dataset = dataset.filter(lambda e: np.any(np.array(e["Labels"])))

# model_path = "roberta-base"
model_path = "roberta-large"
model_path = "results/mlm-large"
model_path = "results/mlm-large_fine-tune/checkpoint-630"


class RobertaForMultilabelSequenceClassification(
    transformers.RobertaForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = torch.nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # (bs, seq_len, hidden_dim)
        pooled_output = outputs[0]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(
                logits,
                labels.float(),
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path, id2label=id2label, label2id=label2id
)
config = transformers.AutoConfig.from_pretrained(model_path)
config.num_labels = len(labels)
model = RobertaForMultilabelSequenceClassification.from_pretrained(
    model_path, config=config
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_data_func(examples):
    batch_encoding = tokenizer(
        text=examples["Premise"],
        text_pair=examples["Conclusion"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    batch_encoding["label"] = torch.tensor(
        [label_list for label_list in examples["Labels"]]
    )
    batch_encoding.to(device)
    return batch_encoding


# def hamming_score(labels, y_pred):
#     assert np.all(np.logical_or(labels, y_pred).sum(axis=1))
#     return (
#         np.logical_and(labels, y_pred).sum(axis=1)
#         / np.logical_or(labels, y_pred).sum(axis=1)
#     ).mean()


# def compute_metrics(eval_pred):
#     """Report Precision, Recall, F1 of among all labels"""
#     # shape: (batch_size, num_labels)
#     logits, labels = eval_pred
#     pred = logits > 0
#     return {
#         "accuracy": hamming_score(labels, pred),
#         "micro_f1": f1_score(labels, pred, average="micro", zero_division=np.nan),
#         "micro_precision": precision_score(
#             labels, pred, average="micro", zero_division=np.nan
#         ),
#         "micro_recall": recall_score(
#             labels, pred, average="micro", zero_division=np.nan
#         ),
#         "macro_f1": f1_score(labels, pred, average="macro", zero_division=np.nan),
#         "macro_precision": precision_score(
#             labels, pred, average="macro", zero_division=np.nan
#         ),
#         "macro_recall": recall_score(
#             labels, pred, average="macro", zero_division=np.nan
#         ),
#     }


dataset = dataset.map(process_data_func, batched=True)


# def save_prediction(trainer, split):
#     predictions = trainer.predict(dataset[split])
#     with open(
#         os.path.join(trainer.args.output_dir, f"{split}_prediction.json"), "w"
#     ) as f:
#         json.dump(
#             [compute_metrics((predictions.predictions, predictions.label_ids))]
#             + [
#                 {
#                     "label_index": np.where(np.array(label_id) > 0)[0].tolist(),
#                     "prediction": np.where(prediction > 0)[0].tolist(),
#                     "logits": prediction.tolist(),
#                     "premise": sample["Premise"],
#                     "conclusion": sample["Conclusion"],
#                     "stance": sample["Stance"],
#                     "label_name": [
#                         id2label[idx] for idx, id in enumerate(label_id) if id != 0
#                     ],
#                 }
#                 for label_id, prediction, sample in zip(
#                     predictions.label_ids.tolist(),
#                     predictions.predictions,
#                     dataset[split],
#                 )[:10]
#             ],
#             f,
#             indent=4,
#             sort_keys=True,
#         )


if __name__ == "__main__":
    fake_batch_size = 256
    real_batch_size = 2
    num_epochs = 30
    eval_steps = len(dataset["train"]) // fake_batch_size
    training_args = transformers.TrainingArguments(
        # output_dir=f"./results/{time.strftime('%Y%m%d_%H%M%S')}",
        output_dir="./results/mlm-large_fine-tune",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=real_batch_size,
        per_device_eval_batch_size=real_batch_size,
        gradient_accumulation_steps=fake_batch_size // real_batch_size,
        weight_decay=0.001,
        logging_dir="./logs",
        fp16=True,
        logging_strategy="epoch",
        max_grad_norm=1.0,
        push_to_hub=False,
        # run_name="MSBD5018 Group Project",
        do_train=True,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        save_total_limit=3,
        metric_for_best_model="micro_f1",
        report_to="wandb",
        run_name="mlm-large_fine-tune",
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
        compute_metrics=partial(compute_metrics, id2label),
    )
    # trainer.train()
    # save_prediction(trainer, "test")
    # save_prediction(trainer, "validation")
    # save_prediction(trainer, "train")
    save_prediction(trainer=trainer, dataset=dataset, split="test", id2label=id2label)
    save_prediction(
        trainer=trainer, dataset=dataset, split="validation", id2label=id2label
    )
    save_prediction(trainer=trainer, dataset=dataset, split="train", id2label=id2label)

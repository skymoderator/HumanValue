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
from transformers import RobertaForMaskedLM
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

model_path = "results/deberta-large_mlm/checkpoint-568"


class MyTrainer(transformers.Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        from transformers.utils.import_utils import is_datasets_available
        import datasets

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        weights = dataset_labels @ torch.tensor(class_weights)
        weights = weights / weights.max()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset["train"]),
            replacement=True,
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": sampler,
        }
        from transformers.trainer_utils import seed_worker

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaPredictionHeadTransform(config)

        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(self.embedding_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        embedding_states = self.transform(hidden_states)
        vocab_logits = self.decoder(embedding_states)
        return vocab_logits, embedding_states


class DebertaForMaskedPromptBasedLM(transformers.DebertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = torch.nn.BCEWithLogitsLoss()
        self.loss_fct2 = torch.nn.BCEWithLogitsLoss(
            weight=torch.tensor(class_weights).to("cuda")
        )
        self.cls.predictions = DebertaLMPredictionHead(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        masked_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        batch_size = input_ids.size(0)
        device = input_ids.device
        # last_hidden_state: (bs, seq_len, hidden_size)
        sequence_output = outputs[0]
        # vocab_logits: (bs, seq_len, vocab_size)
        # embedding_states: (bs, seq_len, embedding_size)
        vocab_logits, embedding_states = self.cls(sequence_output)
        # (bs, vocab_size)
        prediction_scores = vocab_logits[torch.arange(batch_size), masked_position, :]
        # (bs, embedding_size)
        embedding_states = embedding_states[
            torch.arange(batch_size), masked_position, :
        ]
        # (num_labels, embedding_size)
        classes_embedding = self.cls.predictions.decoder.weight[label_token_ids]
        # (bs, num_labels)
        probability = torch.matmul(embedding_states, classes_embedding.T)
        # (bs, )
        # embedding_norm = torch.sqrt(torch.sum(embedding_states**2, dim=1))
        # (num_labels, )
        # classes_embedding_norm = torch.sqrt(torch.sum(classes_embedding**2, dim=1))
        # probability /= embedding_norm.unsqueeze(1) * classes_embedding_norm
        masked_lm_loss = None
        if labels is not None:
            # (bs, 20)
            labels = labels.to(device)
            # convert labels to (bs, vocab_size) where element in indices of labels is 1, else 0
            # (bs, vocab_size)
            one_hotted_labels = torch.zeros_like(prediction_scores)
            # (bs, 20)
            batched_label_token_ids = label_token_ids.unsqueeze(0).repeat(batch_size, 1)
            non_zero_labels_idx = torch.where(labels == 1)
            # (bs, 20)
            batched_label_token_ids = torch.where(
                labels == 1, batched_label_token_ids, labels
            ).long()
            one_hotted_labels[
                non_zero_labels_idx[0], batched_label_token_ids[non_zero_labels_idx]
            ] = 1
            # -100 index = padding token
            masked_lm_loss = self.loss_fct(prediction_scores, one_hotted_labels)
            masked_lm_loss += self.loss_fct2(probability, labels)
        prediction_scores = prediction_scores.cpu()

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        output = MaskedLMOutput(
            loss=masked_lm_loss,
            # logits=prediction_scores,
            logits=probability,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output["probability"] = probability
        return output


def data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = tokenizer.pad(
        features,
        padding="longest",
        return_tensors="pt",
    )
    return batch


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path, id2label=id2label, label2id=label2id
)
config = transformers.AutoConfig.from_pretrained(
    model_path, num_labels=len(labels), id2label=id2label, label2id=label2id
)
model = DebertaForMaskedPromptBasedLM.from_pretrained(model_path, config=config)
# add labels to tokenizer and model
tokenizer.add_tokens(labels, special_tokens=False)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_token_ids = (
    tokenizer(text=labels, add_special_tokens=False, return_tensors="pt")["input_ids"]
    .squeeze()
    .to(device)
)


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
    batch_encoding["labels"] = torch.tensor(examples["Labels"], dtype=torch.float32)
    # batch_encoding["masked_position"] = torch.where(
    #     batch_encoding["input_ids"] == tokenizer.mask_token_id
    # )[1]
    batch_encoding["masked_position"] = torch.tensor(
        [
            [
                i
                for i, token_id in enumerate(ids)
                if token_id == tokenizer.mask_token_id
            ][0]
            for ids in batch_encoding["input_ids"]
        ]
    )
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
        output_dir="./results/deberta-mlm-large_prompt-based-pure-balanced",
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
        run_name="deberta-mlm-large_prompt-based-pure-balanced",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
    )
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        # compute_metrics=lambda z: compute_metrics(
        #     id2label, z, label_token_ids=label_token_ids
        # ),
        compute_metrics=partial(compute_metrics, id2label),
        data_collator=data_collator,
    )
    # trainer.evaluate()
    trainer.train()
    # save_prediction(trainer=trainer, dataset=dataset, split="test", id2label=id2label)
    # save_prediction(
    #     trainer=trainer, dataset=dataset, split="validation", id2label=id2label
    # )
    # save_prediction(trainer=trainer, dataset=dataset, split="train", id2label=id2label)

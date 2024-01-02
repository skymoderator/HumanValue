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

# model_path = "roberta-large"
model_path = "./results/mlm-large"
model_path = "results/mlm-large_prompt-based/checkpoint-168"


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


class RobertaForMaskedPromptBasedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(class_weights))
        self.lm_head.decoder = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.lm_head.bias = torch.nn.Parameter(torch.zeros(config.num_labels))
        self.lm_head.decoder.bias = self.lm_head.bias

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        masked_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # BaseModelOutputWithPoolingAndCrossAttentions
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # last_hidden_state: (bs, seq_len, hidden_size)
        sequence_output = outputs[0]
        # (bs, seq_len, num_labels)
        prediction_scores = self.lm_head(sequence_output)
        # (bs, num_labels)
        prediction_scores = prediction_scores[
            torch.arange(prediction_scores.size(0)), masked_position, :
        ]

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            masked_lm_loss = self.loss_fct(prediction_scores, labels)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path, id2label=id2label, label2id=label2id
)
config = transformers.AutoConfig.from_pretrained(
    model_path, num_labels=len(labels), id2label=id2label, label2id=label2id
)
# model = RobertaForMaskedPromptBasedLM.from_pretrained(model_path, config=config)
config.tie_word_embeddings = False
model = RobertaForMaskedPromptBasedLM.from_pretrained(
    model_path, local_files_only=True, config=config
)
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
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    batch_encoding["labels"] = torch.tensor(examples["Labels"], dtype=torch.float32)
    batch_encoding["masked_position"] = torch.where(
        batch_encoding["input_ids"] == tokenizer.mask_token_id
    )[1]
    batch_encoding.to(device)
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
        output_dir="./results/mlm-large_prompt-based-balanced-thresholding",
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
        # report_to="wandb",
        report_to="none",
        run_name="mlm-large_prompt-based-balanced",
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
        compute_metrics=partial(threshold_metrics, id2label),
    )
    save_threshold_prediction(trainer=trainer, dataset=dataset, id2label=id2label)

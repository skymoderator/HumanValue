import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import torch


def hamming_score(labels, y_pred):
    assert np.all(np.logical_or(labels, y_pred).sum(axis=1))
    return (
        np.logical_and(labels, y_pred).sum(axis=1)
        / np.logical_or(labels, y_pred).sum(axis=1)
    ).mean()


def compute_fs_metrics(
    id2label,
    eval_pred,
):
    """Report Precision, Recall, F1 of among all labels"""
    # logits: (dataset_size, num_labels, 2)
    # labels: (dataset_size, num_labels)
    pred, labels = eval_pred
    # logits = logits[0]
    # (dataset_size, num_labels)
    pred = np.argmax(pred, axis=2)
    official_f1_scores = {
        id2label[i]: round(f1_score(labels[:, i], pred[:, i], zero_division=0), 2)
        for i in range(len(labels[1]))
    }
    return {
        "accuracy": hamming_score(labels, pred),
        "micro_f1": f1_score(labels, pred, average="micro", zero_division=np.nan),
        "micro_precision": precision_score(
            labels, pred, average="micro", zero_division=np.nan
        ),
        "micro_recall": recall_score(
            labels, pred, average="micro", zero_division=np.nan
        ),
        "macro_f1": f1_score(labels, pred, average="macro", zero_division=np.nan),
        "macro_precision": precision_score(
            labels, pred, average="macro", zero_division=np.nan
        ),
        "macro_recall": recall_score(
            labels, pred, average="macro", zero_division=np.nan
        ),
        "official_calculation": {
            "avg-f1-score": round(np.mean(list(official_f1_scores.values())), 2),
            **official_f1_scores,
        },
    }


def compute_metrics(id2label, eval_pred, label_token_ids=None):
    """Report Precision, Recall, F1 of among all labels"""
    if label_token_ids is not None:
        if isinstance(label_token_ids, torch.Tensor):
            label_token_ids = label_token_ids.cpu().numpy()
        # logits: (dataset_size, vocab_size)
        # labels: (dataset_size, num_labels)
        logits, labels = eval_pred
        num_labels = len(id2label)
        # obtain the index of top num_labels logits
        # (dataset_size, num_labels)
        top_idx = np.argpartition(logits, -num_labels, axis=1)[:, -num_labels:]
        # (dataset_size, num_labels)
        labels_extended = labels * label_token_ids
        # (dataset_size, num_labels, num_labels)
        labels_extended = np.repeat(
            np.expand_dims(labels_extended, 2), num_labels, axis=2
        )
        # (dataset_size, num_labels, 20)
        pred = labels_extended - np.expand_dims(top_idx, 1)
        # (dataset_size, num_labels)
        pred = np.any(pred == 0, axis=2)
        pred *= labels.astype(np.bool_)
    else:
        # shape: (dataset_size, num_labels)
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            prob = 1 / (1 + np.exp(-logits[0]))
            pred = prob >= 0.5
        else:
            pred = logits > 0
    official_f1_scores = {
        id2label[i]: round(f1_score(labels[:, i], pred[:, i], zero_division=0), 2)
        for i in range(len(labels[1]))
    }
    return {
        "accuracy": hamming_score(labels, pred),
        "micro_f1": f1_score(labels, pred, average="micro", zero_division=np.nan),
        "micro_precision": precision_score(
            labels, pred, average="micro", zero_division=np.nan
        ),
        "micro_recall": recall_score(
            labels, pred, average="micro", zero_division=np.nan
        ),
        "macro_f1": f1_score(labels, pred, average="macro", zero_division=np.nan),
        "macro_precision": precision_score(
            labels, pred, average="macro", zero_division=np.nan
        ),
        "macro_recall": recall_score(
            labels, pred, average="macro", zero_division=np.nan
        ),
        "official_calculation": {
            "avg-f1-score": round(np.mean(list(official_f1_scores.values())), 2),
            **official_f1_scores,
        },
    }


def threshold_metrics(id2label, eval_pred, thresholds=None):
    # shape: (dataset_size, num_labels)
    logits, labels = eval_pred
    prob = 1 / (1 + np.exp(-logits))
    # (num_labels, )
    if thresholds is None:
        thresholds = []
        best_f1_scores = []
        num_labels = len(id2label)
        for i in range(num_labels):
            best_f1_score = 0
            best_threshold = 0
            for threshold in np.linspace(0, 1, 100):
                pred = prob[:, i] >= threshold
                f1 = f1_score(labels[:, i], pred, zero_division=0)
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_threshold = threshold
            thresholds.append(best_threshold)
            best_f1_scores.append(best_f1_score)
        print(f"thresholds: {thresholds}")
        print(f"best_f1_scores: {best_f1_scores}")
        return thresholds
    pred = prob >= thresholds
    official_f1_scores = {
        id2label[i]: round(f1_score(labels[:, i], pred[:, i], zero_division=0), 2)
        for i in range(len(labels[1]))
    }
    return {
        "accuracy": hamming_score(labels, pred),
        "micro_f1": f1_score(labels, pred, average="micro", zero_division=np.nan),
        "micro_precision": precision_score(
            labels, pred, average="micro", zero_division=np.nan
        ),
        "micro_recall": recall_score(
            labels, pred, average="micro", zero_division=np.nan
        ),
        "macro_f1": f1_score(labels, pred, average="macro", zero_division=np.nan),
        "macro_precision": precision_score(
            labels, pred, average="macro", zero_division=np.nan
        ),
        "macro_recall": recall_score(
            labels, pred, average="macro", zero_division=np.nan
        ),
        "official_calculation": {
            "avg-f1-score": round(np.mean(list(official_f1_scores.values())), 2),
            **official_f1_scores,
        },
    }


def save_threshold_prediction(trainer, dataset, id2label):
    eval_predictions = trainer.predict(dataset["validation"].remove_columns("labels"))
    thresholds = threshold_metrics(
        id2label,
        (eval_predictions.predictions, np.array(dataset["validation"]["labels"])),
    )
    predictions = trainer.predict(dataset["test"].remove_columns("labels"))
    metrics = threshold_metrics(
        id2label,
        (predictions.predictions, np.array(dataset["test"]["labels"])),
        thresholds,
    )
    with open(
        os.path.join(trainer.args.output_dir, f"threshold_prediction.json"), "w"
    ) as f:
        json.dump(
            [metrics],
            f,
            indent=4,
            sort_keys=True,
        )


def save_prediction(trainer, dataset, split, id2label):
    predictions = trainer.predict(dataset[split])
    with open(
        os.path.join(trainer.args.output_dir, f"{split}_prediction.json"), "w"
    ) as f:
        json.dump(
            [predictions.metrics]
            + [
                {
                    "label_index": np.where(np.array(label_id) > 0)[0].tolist(),
                    "prediction": np.where(prediction > 0)[0].tolist(),
                    "prediction_name": [
                        id2label[idx] for idx in np.where(prediction > 0)[0].tolist()
                    ],
                    "logits": prediction.tolist(),
                    "premise": dataset[split][index]["Premise"],
                    "conclusion": dataset[split][index]["Conclusion"],
                    "stance": dataset[split][index]["Stance"],
                    "label_name": [
                        id2label[idx] for idx, id in enumerate(label_id) if id != 0
                    ],
                }
                for index, (label_id, prediction) in enumerate(
                    zip(
                        predictions.label_ids.tolist()[:10],
                        predictions.predictions[:10],
                    )
                )
            ],
            f,
            indent=4,
            sort_keys=True,
        )

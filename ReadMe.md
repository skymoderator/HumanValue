# Humman Value Detection
This project is an attempt for the [SemEval 2023 Task 4 competition](https://touche.webis.de/clef24/touche24-web/human-value-detection.html) competition. The goal of this competition is to predict the human values of a given text. The human values are defined as the following:

- Self-direction: thought
- Self-direction: action
- Stimulation
- Hedonism
- Achievement
- Power: dominance
- Power: resources
- Face
- Security: personal
- Security: societal
- Tradition
- Conformity: rules
- Conformity: interpersonal
- Humility
- Benevolence: caring
- Benevolence: dependability
- Universalism: concern
- Universalism: nature
- Universalism: tolerance
- Universalism: objectivity

Full codes, paper, models, model weights and datasets are available [here](https://drive.google.com/drive/folders/15DjN7JGAlt0lAFWt8gdjAjLCHaM5ZMGn?usp=drive_link).

## Introduction
In this project, I conducted abalation studies on different model settings (hyperparameters, model architecture, and training setting) to seek for the best model.

The best model I found is the RoBERTa-Large model undergoing MLM training on the training set and fine-tuned with the prompt-based setting, which achieves 0.59 F1 score on the test set.

## Code & Model Setting
The main.py is responsible for the experiements (see the Table 3 of my report P.5) of:
1. RoBERTa-base baseline (set model_path='result\fine-tune\checkpoint-630' in line 75)
2. RoBERTa-base MLM + F.T. (set model_path='result\mlm_fine-tune\checkpoint-609' in line 75)
3. RoBERTa-large MLM + F.T. (set model_path='result\mlm-large_fine-tune\checkpoint-609' in line 75)

The mlm-prompt.py is responsible for the experiements of:
1. RoBERTa-Large MLM + Prompt-based (set model_path='result\mlm-large_prompt-based\checkpoint-168' in line 102)
2. RoBERTa-Large MLM + Prompt-based balanced (set model_path='result\mlm-large_prompt-based-balanced\checkpoint-105' in line 102)

The deberta-mlm-prompt.py is responsible for the experiements of:
1. DeBERTa-Large Prompt-based balanced (set model_path='result\deberta-large_prompt-based-balanced\checkpoint-168' in line 108)
2. DeBERTa-Large MLM Prompt-based balanced (set model_path='result\deberta-mlm-large_prompt-based-balanced\checkpoint-21' in line 108)

Finally, for the number reported on the report Table 3, see the result\(model setting)\test_prediction.json file.

## Side Note

Since each .py file (main.py, mlm-prompt.py, and deberta-mlm-prompt.py) is responsible for different experiments, if you want to run the code (for both train and eval), you need to change the output_dir in the code to ensure that the final model weights and predictions.json are saved in the correct folder. The output_dir can be set in the transformers.TrainingArgument defined in the main.py, mlm-prompt.py and deberta-mlm-prompt.py, respectively.
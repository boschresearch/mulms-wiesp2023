# Experiment resources related to the MuLMS corpus (WIESP 2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
This module contains the training pipeline for BILOU tagging-based NER. It uses a BERT-based LM in combination with a conditional random field (CRF).
"""

import logging
import os
import pickle
import sys
from argparse import ArgumentParser
from copy import deepcopy

import pandas as pd
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.constants.mulms_constants import (
    CPU,
    mspt_bilou_ne_labels,
    sofc_bilou_ne_labels,
)
from source.constants.nested_bilou_tags import mulms_all_bilou_tags
from source.ner.datasets.mspt_ner_dataset import MSPT_NER_Dataset
from source.ner.datasets.ner_bilou_dataset import MULMS_NER_BILOU_Dataset
from source.ner.datasets.sofc_ner_dataset import SOFC_NER_Dataset
from source.ner.evaluation.bilou_pretty_print import print_bilou_to_token_matchings
from source.ner.evaluation.metrics import (
    calculate_prec_recall_f1_for_bilou,
    calculate_prec_recall_f1_for_bio,
)
from source.ner.models.bert_crf import NER_CRF_Classifier
from source.ner.models.multi_task_bert_crf import MT_NER_CRF_Classifier
from source.relation_extraction.data_handling.sofc_relation_dataset import *  # noqa: F401, F403
from source.utils.helper_functions import (
    get_executor_device,
    move_to_device,
    print_cmd_args,
)
from source.utils.lr_scheduler import SqrtSchedule
from source.utils.multitask_dataloader import MultitaskDataloader

parser: ArgumentParser = ArgumentParser()
parser.add_argument("--model_name", type=str, help="Name of/Path to the pretrained model")
parser.add_argument(
    "--output_path", type=str, help="Storage path for fine-tuned model", default="."
)
parser.add_argument(
    "--disable_model_storage",
    action="store_true",
    help="Whether to disable storage of final model",
)
parser.add_argument(
    "--disable_cuda", action="store_true", help="Disable CUDA in favour of CPU usage"
)
parser.add_argument("--seed", type=int, help="Random seed", default=23081861)
parser.add_argument("--batch_size", type=int, help="Batch size used during training", default=32)
parser.add_argument("--lr", type=float, help="Learning rate of the model head", default=5e-5)
parser.add_argument("--lr_crf", type=float, help="Learning rate of the CRF layer", default=1e-3)
parser.add_argument("--num_epochs", type=int, help="Number of training epochs", default=100)
parser.add_argument(
    "--cv",
    type=int,
    help="If set, the corresponding train set is used as tune set for CV training.",
    choices=[1, 2, 3, 4, 5],
    default=None,
)
parser.add_argument(
    "--enable_lr_decay", action="store_true", help="Whether to use learning rate decay"
)
parser.add_argument(
    "--enable_bilou_pretty_print",
    action="store_true",
    help="If true, evaluation will print all tokens and tags with color encoding to see correct and incorrect predictions.",
)
parser.add_argument(
    "--write_per_sample_predictions_to_file",
    action="store_true",
    help="Whether to store predictions for each sample as Pickle dict and CSV file.",
)
parser.add_argument(
    "--main_single_task_dataset",
    choices=["MULMS", "SOFC", "MSPT"],
    type=str,
    help="This dataset will be used for NER single-task training.",
    default="MULMS",
)
parser.add_argument(
    "--enable_multi_tasking",
    action="store_true",
    help="Whether to enable multi-task training with SOFC and/or MSPT named entities.",
)
parser.add_argument(
    "--main_mt_dataset",
    type=str,
    choices=["MULMS", "SOFC", "MSPT"],
    help="If multi-tasking is enabled, this dataset is used for evaluation, i.e., the focus lies upon this dataset.",
    default="MULMS",
)
parser.add_argument(
    "--add_mt_dataset",
    type=str,
    choices=["MULMS", "SOFC", "MSPT", "ALL"],
    help="Which NER dataset to additionally use for multi-tasking. MUST NOT be the same as --main_mt_dataset",
    default="SOFC",
)
parser.add_argument(
    "--split_mt_optimizers",
    action="store_true",
    help="Whether to use different optimizers for different output layers when using multi-tasking. (only hardcoded additional learning rates are supported)",
)

args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

DEVICE: str = get_executor_device(disable_cuda=args.disable_cuda)


def train(
    classifier: NER_CRF_Classifier,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    val_gt_ne_annots: list[set[tuple]],
    lm_optimizer: "AdamW | list[AdamW]",
    crf_optimizer: "AdamW | list[AdamW]",
    lm_scheduler: "LambdaLR | list[LambdaLR]",
    crf_scheduler: "LambdaLR | list[LambdaLR]",
) -> NER_CRF_Classifier:
    """
    Executes the training loop and fine-tunes a CRF-based classifier.

    Args:
        classifier (NER_CRF_Classifier): Trainable classifier
        dataloader (DataLoader): MuLMS dataset loader
        val_dataloader (DataLoader): Validation data for early stopping
        val_gt_ne_annots (list[set[tuple]]): GT labels for validation set
        lm_optimizer (AdamW | list[AdamW]): (List of) BERT optimizer
        crf_optimizer (AdamW | list[AdamW]): (List of) CRF optimizer
        lm_scheduler (LambdaLR | list[LambdaLR]): (List of) BERT learning rate scheduler
        crf_scheduler (LambdaLR | list[LambdaLR]): (List of) CRF learning rate scheduler

    Returns:
        NER_CRF_Classifier: The fine-tuned NER classifier
    """
    best_model: NER_CRF_Classifier = deepcopy(classifier)
    best_f1: float = 0.0
    early_stopping_counter: int = 0

    for i in range(args.num_epochs):
        logging.log(logging.INFO, f"Starting epoch {i+1}/{args.num_epochs}")
        classifier = classifier.train()

        for batch in tqdm(dataloader):
            if DEVICE != CPU:
                (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                    batch["label_ids"],
                ) = move_to_device(
                    DEVICE,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                    batch["label_ids"],
                )

            loss: torch.Tensor = None

            if "dataset" in batch:
                loss = -1 * classifier.get_loss(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                    batch["label_ids"],
                    batch["dataset"],
                )
            else:
                loss = -1 * classifier.get_loss(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                    batch["label_ids"],
                )

            loss.backward()
            if type(lm_optimizer) == list:
                for opt in lm_optimizer:
                    opt.step()
                for opt in crf_optimizer:
                    opt.step()
            else:
                lm_optimizer.step()
                crf_optimizer.step()

            if type(lm_optimizer) == list:
                for opt in lm_optimizer:
                    opt.zero_grad()
                for opt in crf_optimizer:
                    opt.zero_grad()
            else:
                lm_optimizer.zero_grad()
                crf_optimizer.zero_grad()

            if args.enable_lr_decay:
                if type(lm_scheduler) == list:
                    for s in lm_scheduler:
                        s.step()
                    for c in crf_scheduler:
                        c.step()
                else:
                    lm_scheduler.step()
                    crf_scheduler.step()

            if DEVICE != CPU:
                (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                    batch["label_ids"],
                ) = move_to_device(
                    CPU,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                    batch["label_ids"],
                )

        logging.log(logging.INFO, "Finished epoch training. Starting evaluation.")

        eval_scores: dict = evaluate(
            classifier=classifier, dataloader=val_dataloader, gt_ne_annots=val_gt_ne_annots
        )

        logging.log(
            logging.INFO,
            f"Current Micro F1: {eval_scores['Micro_F1']}, Current Macro F1: {eval_scores['Macro_F1']}",
        )

        if eval_scores["Micro_F1"] > best_f1:
            best_f1 = eval_scores["Micro_F1"]
            best_model = deepcopy(classifier)
            early_stopping_counter: int = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter == 5:
            break

    return best_model


def evaluate(
    classifier: NER_CRF_Classifier,
    dataloader: DataLoader,
    gt_ne_annots: list[set[tuple]],
    pretty_print_bilou: bool = False,
    return_predictions_per_sample: bool = False,
    default_dataset_flag: int = 0,
) -> dict:
    """
    Evaluates the NER classifier.

    Args:
        classifier (NER_CRF_Classifier): Fine-tuned CRF-based NER classifier
        dataloader (DataLoader): Evaluation data
        gt_ne_annots (list[set[tuple]]): GT labels
        pretty_print_bilou (bool, optional): If enabled, the BILOU predictions will be printed against the ground truth tag sequence. Allows for better inspection. Defaults to False.
        return_predictions_per_sample (bool, optional): If enabled, per-sample tag predictions will be returned.. Defaults to False.
        default_dataset_flag (int, optional): Indicates the position of the target dataset in a multi-task scenario. Defaults to 0.

    Returns:
        dict: Computed metrics (P, R, F1)
    """
    classifier = classifier.eval()
    predicted_tag_sequences: list[list] = []
    for batch in dataloader:
        if DEVICE != CPU:
            (
                batch["input_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
                batch["crf_mask"],
                batch["sorted_crf_mask"],
            ) = move_to_device(
                DEVICE,
                batch["input_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
                batch["crf_mask"],
                batch["sorted_crf_mask"],
            )

        pred = classifier.predict_tag_sequence(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["crf_mask"],
            batch["sorted_crf_mask"],
            default_dataset_flag,
        )
        predicted_tag_sequences.extend(pred)

        if DEVICE != CPU:
            (
                batch["input_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
                batch["crf_mask"],
                batch["sorted_crf_mask"],
            ) = move_to_device(
                CPU,
                batch["input_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
                batch["crf_mask"],
                batch["sorted_crf_mask"],
            )
    predicted_tag_sequences = [
        [dataloader.dataset.bilou_id2ne_label[i] for i in p] for p in predicted_tag_sequences
    ]
    if pretty_print_bilou:
        print_bilou_to_token_matchings(
            dataloader.dataset.get_token_list(),
            predicted_tag_sequences,
            dataloader.dataset.get_gold_labels()[1],
        )

    if (not args.enable_multi_tasking and args.main_single_task_dataset == "SOFC") or (
        args.enable_multi_tasking and args.main_mt_dataset == "SOFC"
    ):
        return calculate_prec_recall_f1_for_bio(
            predicted_tag_sequences,
            gt_ne_annots,
            dataloader.dataset.ne_labels,
            return_predictions_per_sample,
        )
    else:
        return calculate_prec_recall_f1_for_bilou(
            predicted_tag_sequences,
            gt_ne_annots,
            dataloader.dataset.ne_labels,
            return_predictions_per_sample,
        )


# Used when enabling additional MT learning rates
predefined_learning_rates: dict = {
    "MULMS": {"LM": 1e-4, "CRF": 7e-3},
    "SOFC": {"LM": 3e-4, "CRF": 7e-3},
    "MSPT": {"LM": 5e-5, "CRF": 9e-3},
}


def main():
    """
    Entry point.
    """
    print_cmd_args(args=args)
    torch.manual_seed(args.seed)
    enable_tune: bool = args.cv is not None

    # If multi-task focus in on another dataset, don't use tune splits in MuLMS
    if args.enable_multi_tasking:
        assert (
            args.main_mt_dataset != args.add_mt_dataset
        ), "At least two different datasets are required for multi-task training."
        if args.main_mt_dataset != "MULMS":
            enable_tune = False

    output_path: str = None
    if args.cv is None:
        # To make it work with evaluation scripts, we artificially create "cv_" output directories
        existing_cv_dirs: list[str] = [
            int(d.split("_")[-1]) for d in os.listdir(args.output_path) if "cv_" in d
        ]
        current_cv: int = max(existing_cv_dirs) + 1 if len(existing_cv_dirs) > 0 else 1
        output_path = os.path.join(args.output_path, f"cv_{current_cv}")
    else:
        output_path = os.path.join(args.output_path, f"cv_{args.cv}")

    os.makedirs(output_path, exist_ok=True)

    # Used if single-task training is desired
    train_ner_dataset = None
    tune_ner_dataset = None
    dev_ner_dataset = None
    test_ner_dataset = None

    # To keep it simple, we first read all datasets despite of maybe not needing all

    # MuLMS datasets
    if enable_tune:
        mulms_train_ner_dataset: MULMS_NER_BILOU_Dataset = MULMS_NER_BILOU_Dataset(
            split="train", tokenizer_model_name=args.model_name, tune_id=args.cv
        )
        mulms_tune_ner_dataset: MULMS_NER_BILOU_Dataset = MULMS_NER_BILOU_Dataset(
            split="tune", tokenizer_model_name=args.model_name, tune_id=args.cv
        )
    else:
        mulms_train_ner_dataset: MULMS_NER_BILOU_Dataset = MULMS_NER_BILOU_Dataset(
            split="train", tokenizer_model_name=args.model_name
        )
    mulms_train_and_tune_ner_dataset: MULMS_NER_BILOU_Dataset = MULMS_NER_BILOU_Dataset(
        split="train", tokenizer_model_name=args.model_name
    )
    mulms_dev_ner_dataset: MULMS_NER_BILOU_Dataset = MULMS_NER_BILOU_Dataset(
        split="validation", tokenizer_model_name=args.model_name
    )
    mulms_test_ner_dataset: MULMS_NER_BILOU_Dataset = MULMS_NER_BILOU_Dataset(
        split="test", tokenizer_model_name=args.model_name
    )

    # SOFC datasets
    sofc_train_ner_dataset: SOFC_NER_Dataset = SOFC_NER_Dataset(
        split="train", bert_model_path=args.model_name, entity_type="ENTITY"
    )
    sofc_dev_ner_dataset: SOFC_NER_Dataset = SOFC_NER_Dataset(
        split="validation", bert_model_path=args.model_name, entity_type="ENTITY"
    )
    sofc_test_ner_dataset: SOFC_NER_Dataset = SOFC_NER_Dataset(
        split="test", bert_model_path=args.model_name, entity_type="ENTITY"
    )

    # MSPT datasets
    mspt_train_ner_dataset: MSPT_NER_Dataset = MSPT_NER_Dataset(
        split="ner-train", tokenizer_model_name=args.model_name
    )
    mspt_dev_ner_dataset: MSPT_NER_Dataset = MSPT_NER_Dataset(
        split="ner-dev", tokenizer_model_name=args.model_name
    )
    mspt_test_ner_dataset: MSPT_NER_Dataset = MSPT_NER_Dataset(
        split="ner-test", tokenizer_model_name=args.model_name
    )

    classifier: NER_CRF_Classifier = None
    val_dataloader: DataLoader = None
    mt_dataloader: MultitaskDataloader = None

    # Multi-Task case
    if args.enable_multi_tasking:
        mt_datasets: list = []
        mt_labels: list[list[str]] = []

        # First determine the target dataset of the multi-task training
        if args.main_mt_dataset == "MULMS":
            mt_datasets.append(DataLoader(mulms_train_ner_dataset, batch_size=args.batch_size))
            val_dataloader = DataLoader(
                mulms_tune_ner_dataset, batch_size=args.batch_size
            )  # Only supports training with tune folds here

            train_ner_dataset = mulms_train_ner_dataset
            tune_ner_dataset = mulms_tune_ner_dataset
            dev_ner_dataset = mulms_dev_ner_dataset
            test_ner_dataset = mulms_test_ner_dataset

            _, _, val_gt_ne_annots = mulms_tune_ner_dataset.get_gold_labels()
            mt_labels.append(mulms_all_bilou_tags)

        elif args.main_mt_dataset == "SOFC":
            mt_datasets.append(DataLoader(sofc_train_ner_dataset, batch_size=args.batch_size))
            val_dataloader = DataLoader(sofc_dev_ner_dataset, batch_size=args.batch_size)

            train_ner_dataset = sofc_train_ner_dataset
            tune_ner_dataset = None
            dev_ner_dataset = sofc_dev_ner_dataset
            test_ner_dataset = sofc_test_ner_dataset

            _, _, val_gt_ne_annots = sofc_dev_ner_dataset.get_gold_labels()
            mt_labels.append(sofc_bilou_ne_labels)

        elif args.main_mt_dataset == "MSPT":
            mt_datasets.append(DataLoader(mspt_train_ner_dataset, batch_size=args.batch_size))
            val_dataloader = DataLoader(mspt_dev_ner_dataset, batch_size=args.batch_size)

            train_ner_dataset = mspt_train_ner_dataset
            tune_ner_dataset = None
            dev_ner_dataset = mspt_dev_ner_dataset
            test_ner_dataset = mspt_test_ner_dataset

            _, _, val_gt_ne_annots = mspt_dev_ner_dataset.get_gold_labels()
            mt_labels.append(mspt_bilou_ne_labels)

        # Now add further datasets to multi-task training

        if (
            args.add_mt_dataset == "MULMS" or args.add_mt_dataset == "ALL"
        ) and args.main_mt_dataset != "MULMS":
            mt_datasets.append(
                DataLoader(mulms_train_and_tune_ner_dataset, batch_size=args.batch_size)
            )
            mt_labels.append(mulms_all_bilou_tags)

        if (
            args.add_mt_dataset == "SOFC" or args.add_mt_dataset == "ALL"
        ) and args.main_mt_dataset != "SOFC":
            mt_datasets.append(DataLoader(sofc_train_ner_dataset, batch_size=args.batch_size))
            mt_labels.append(sofc_bilou_ne_labels)

        if (
            args.add_mt_dataset == "MSPT" or args.add_mt_dataset == "ALL"
        ) and args.main_mt_dataset != "MSPT":
            mt_datasets.append(DataLoader(mspt_train_ner_dataset, batch_size=args.batch_size))
            mt_labels.append(mspt_bilou_ne_labels)

        mt_dataloader = MultitaskDataloader(*mt_datasets)
        classifier = MT_NER_CRF_Classifier(args.model_name, mt_labels).to(DEVICE)

    # Single task case
    else:
        # Target MuLMS
        if args.main_single_task_dataset == "MULMS":
            train_ner_dataset = mulms_train_ner_dataset
            tune_ner_dataset = mulms_tune_ner_dataset
            dev_ner_dataset = mulms_dev_ner_dataset
            test_ner_dataset = mulms_test_ner_dataset
            classifier = NER_CRF_Classifier(args.model_name, mulms_all_bilou_tags).to(DEVICE)

        # Target SOFC
        elif args.main_single_task_dataset == "SOFC":
            train_ner_dataset = sofc_train_ner_dataset
            tune_ner_dataset = None
            dev_ner_dataset = sofc_dev_ner_dataset
            test_ner_dataset = sofc_test_ner_dataset
            classifier = NER_CRF_Classifier(args.model_name, sofc_bilou_ne_labels).to(DEVICE)

        # Target MSPT
        elif args.main_single_task_dataset == "MSPT":
            train_ner_dataset = mspt_train_ner_dataset
            tune_ner_dataset = None
            dev_ner_dataset = mspt_dev_ner_dataset
            test_ner_dataset = mspt_test_ner_dataset
            classifier = NER_CRF_Classifier(args.model_name, mspt_bilou_ne_labels).to(DEVICE)

    train_dataloader: DataLoader = DataLoader(
        train_ner_dataset, batch_size=args.batch_size, shuffle=True
    )
    dev_dataloader: DataLoader = DataLoader(dev_ner_dataset, batch_size=args.batch_size)
    test_dataloader: DataLoader = DataLoader(test_ner_dataset, batch_size=args.batch_size)

    if tune_ner_dataset:
        tune_dataloader: DataLoader = DataLoader(tune_ner_dataset, batch_size=args.batch_size)
        val_dataloader = tune_dataloader
        _, _, val_gt_ne_annots = tune_ner_dataset.get_gold_labels()
    else:
        val_dataloader = dev_dataloader
        _, _, val_gt_ne_annots = dev_ner_dataset.get_gold_labels()

    if args.enable_multi_tasking and args.split_mt_optimizers:
        lm_optimizer: list[AdamW] = [
            AdamW(classifier.get_lm_parameters_only(), lr=args.lr),
            AdamW(classifier.get_linear_layer_parameters_by_index(0), lr=args.lr),
            AdamW(
                classifier.get_linear_layer_parameters_by_index(1),
                lr=predefined_learning_rates[args.add_mt_dataset]["LM"],
            ),
        ]
        crf_optimizer: list[AdamW] = [
            AdamW(classifier.get_crf_layer_parameters_by_index(0), lr=args.lr_crf),
            AdamW(
                classifier.get_crf_layer_parameters_by_index(1),
                lr=predefined_learning_rates[args.add_mt_dataset]["CRF"],
            ),
        ]
    else:
        lm_optimizer: AdamW = AdamW(classifier.get_lm_parameters(), lr=args.lr)
        crf_optimizer: AdamW = AdamW(classifier.get_crf_parameters(), lr=args.lr_crf)

    lm_scheduler = None
    crf_scheduler = None

    if args.enable_lr_decay:
        if args.enable_multi_tasking and args.split_mt_optimizers:
            lm_scheduler: list[LambdaLR] = [
                LambdaLR(optimizer=opt, lr_lambda=SqrtSchedule(len(train_dataloader)))
                for opt in lm_optimizer
            ]
            crf_scheduler: list[LambdaLR] = [
                LambdaLR(optimizer=opt, lr_lambda=SqrtSchedule(len(train_dataloader)))
                for opt in crf_optimizer
            ]
        else:
            lm_scheduler: LambdaLR = LambdaLR(
                optimizer=lm_optimizer, lr_lambda=SqrtSchedule(len(train_dataloader))
            )
            crf_scheduler: LambdaLR = LambdaLR(
                optimizer=crf_optimizer, lr_lambda=SqrtSchedule(len(train_dataloader))
            )

    # Start Training

    best_model: NER_CRF_Classifier = train(
        classifier,
        mt_dataloader if mt_dataloader is not None else train_dataloader,
        val_dataloader,
        val_gt_ne_annots,
        lm_optimizer,
        crf_optimizer,
        lm_scheduler,
        crf_scheduler,
    )

    _, _, gt_ne_annots_dev = dev_ner_dataset.get_gold_labels()
    _, _, gt_ne_annots_test = test_ner_dataset.get_gold_labels()
    final_dev_scores: dict = None
    final_test_scores: dict = None
    if args.write_per_sample_predictions_to_file:
        final_dev_scores, predictions_dev = evaluate(
            best_model,
            dev_dataloader,
            gt_ne_annots=gt_ne_annots_dev,
            pretty_print_bilou=args.enable_bilou_pretty_print,
            return_predictions_per_sample=args.write_per_sample_predictions_to_file,
        )
        dev_tokens: list[list[str]] = dev_ner_dataset._token_list
        dev_predictions: list[dict] = []

        for tokens, preds in zip(dev_tokens, predictions_dev):
            tmp_dict: dict = {
                "tokens": tokens,
                "text": [],
                "id": list(range(1, len(preds) + 1)),
                "value": [],
                "tokenIndices": [],
            }
            for p in preds:
                tmp_dict["text"].append(" ".join(tokens[p[1] : p[2] + 1]))
                tmp_dict["value"].append(p[0])
                tmp_dict["tokenIndices"].append([p[1], p[2]])
            dev_predictions.append(tmp_dict)

        with open(os.path.join(output_path, "predictions_dev.pickle"), "wb") as f:
            pickle.dump(dev_predictions, f)
        dev_predictions_df: pd.DataFrame = pd.DataFrame.from_dict(dev_predictions)
        with open(os.path.join(output_path, "predictions_dev.csv"), "wb") as f:
            dev_predictions_df.to_csv(f)

        final_test_scores, predictions_test = evaluate(
            best_model,
            test_dataloader,
            gt_ne_annots=gt_ne_annots_test,
            pretty_print_bilou=args.enable_bilou_pretty_print,
            return_predictions_per_sample=args.write_per_sample_predictions_to_file,
        )
        test_tokens: list[list[str]] = test_ner_dataset._token_list
        test_predictions: list[dict] = []
        for tokens, preds in zip(test_tokens, predictions_test):
            tmp_dict: dict = {
                "tokens": tokens,
                "text": [],
                "id": list(range(1, len(preds) + 1)),
                "value": [],
                "tokenIndices": [],
            }
            for p in preds:
                tmp_dict["text"].append(" ".join(tokens[p[1] : p[2] + 1]))
                tmp_dict["value"].append(p[0])
                tmp_dict["tokenIndices"].append([p[1], p[2]])
            test_predictions.append(tmp_dict)

        with open(os.path.join(output_path, "predictions_test.pickle"), "wb") as f:
            pickle.dump(test_predictions, f)
        test_predictions_df: pd.DataFrame = pd.DataFrame.from_dict(test_predictions)
        with open(os.path.join(output_path, "predictions_test.csv"), "wb") as f:
            test_predictions_df.to_csv(f)

    else:
        final_dev_scores = evaluate(
            best_model,
            dev_dataloader,
            gt_ne_annots=gt_ne_annots_dev,
            pretty_print_bilou=args.enable_bilou_pretty_print,
        )
        final_test_scores = evaluate(
            best_model,
            test_dataloader,
            gt_ne_annots=gt_ne_annots_test,
            pretty_print_bilou=args.enable_bilou_pretty_print,
        )

    if not args.disable_model_storage:
        torch.save(best_model.state_dict(), os.path.join(output_path, "ner_bilou_tagger.pt"))

    with open(os.path.join(output_path, "scores_dev.pickle"), mode="wb") as f:
        pickle.dump(final_dev_scores, f)
    with open(os.path.join(output_path, "scores_test.pickle"), mode="wb") as f:
        pickle.dump(final_test_scores, f)
    return


if __name__ == "__main__":
    main()

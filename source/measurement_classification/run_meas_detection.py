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
This module contains the training pipeline for the Measurement Classification task.
"""

import logging
import os
import pickle
import sys
from argparse import ArgumentParser
from copy import deepcopy

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.constants.mulms_constants import CPU, CUDA, id2meas_label, meas_labels
from source.measurement_classification.datasets.measurement_dataset import (
    MeasurementDataset,
)
from source.measurement_classification.models.measurement_classifier import (
    MeasurementClassifier,
    RandomMeasurementClassifier,
)
from source.utils.helper_functions import move_to_device, print_cmd_args
from source.utils.lr_scheduler import SqrtSchedule

parser: ArgumentParser = ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    help="Name of/Path to the pretrained model",
    default="allenai/scibert_scivocab_uncased",
)
parser.add_argument(
    "--output_path", type=str, help="Storage path for fine-tuned model", default="."
)
parser.add_argument(
    "--disable_model_storage",
    help="Disables storage of model parameters in order to save disk space.",
    action="store_true",
)
parser.add_argument("--seed", type=int, help="Random seed", default=1103)
parser.add_argument(
    "--disable_cuda", action="store_true", help="Disable CUDA in favour of CPU usage"
)
parser.add_argument("--lr", type=float, help="Learning rate", default=5e-6)
parser.add_argument("--batch_size", type=int, help="Batch size used during training", default=32)
parser.add_argument("--num_epochs", type=int, help="Number of training epochs", default=100)
parser.add_argument("--dropout_rate", type=float, help="Dropout rate during training", default=0.1)
parser.add_argument(
    "--cv",
    type=int,
    help="If set, the corresponding train set is used as tune set for CV training.",
    choices=[1, 2, 3, 4, 5],
    default=None,
)
parser.add_argument(
    "--include_non_meas_sents",
    action="store_true",
    help="Whether to include sentences that are not annotated with a measurement span.",
)
parser.add_argument(
    "--disable_lr_decay", action="store_true", help="Whether to not use learning rate decay"
)
parser.add_argument(
    "--subsample_rate",
    type=float,
    help="Percentage by which to subsample non-measurement sentences",
    default=1.0,
)
parser.add_argument(
    "--use_random_classifier",
    action="store_true",
    help="If set, a random classifier with priors estimated from the train set will be used instead of a deep-learning based model.",
)

args = parser.parse_args()

DEVICE: str = CUDA if torch.cuda.is_available() and not args.disable_cuda else CPU
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def evaluate(classifier: MeasurementClassifier, dataloader: DataLoader) -> dict:
    """
    Evaluates the trained model on measurement classification.

    Args:
        classifier (MeasurementClassifier): The fine-tuned measurement classifier.
        dataloader (DataLoader): The dataloader that yields the evaluation samples.

    Returns:
        dict: Computed scores
    """
    classifier = classifier.eval()
    predictions: list[int] = []
    gt_labels: list[int] = dataloader.dataset._data["label_ids"]
    for batch in dataloader:
        if DEVICE != CPU:
            (
                batch["tensor"]["input_ids"],
                batch["tensor"]["attention_mask"],
                batch["tensor"]["token_type_ids"],
                batch["label"],
            ) = move_to_device(
                DEVICE,
                batch["tensor"]["input_ids"],
                batch["tensor"]["attention_mask"],
                batch["tensor"]["token_type_ids"],
                batch["label"],
            )
        logits: torch.Tensor = classifier(
            batch["tensor"]["input_ids"],
            batch["tensor"]["attention_mask"],
            batch["tensor"]["token_type_ids"],
        )
        predictions.extend(torch.argmax(logits, axis=1).tolist())

        if DEVICE != CPU:
            (
                batch["tensor"]["input_ids"],
                batch["tensor"]["attention_mask"],
                batch["tensor"]["token_type_ids"],
                batch["label"],
            ) = move_to_device(
                CPU,
                batch["tensor"]["input_ids"],
                batch["tensor"]["attention_mask"],
                batch["tensor"]["token_type_ids"],
                batch["label"],
            )
    P, R, F, S = precision_recall_fscore_support(gt_labels, predictions)
    scores: dict = {
        id2meas_label[id]: {"P": P[id], "R": R[id], "F": F[id], "S": S[id]}
        for id in set(predictions)
    }
    micro_P, micro_R, micro_F, _ = precision_recall_fscore_support(
        gt_labels, predictions, average="micro"
    )
    scores["Micro_P"] = micro_P
    scores["Micro_R"] = micro_R
    scores["Micro_F1"] = micro_F

    macro_P, macro_R, macro_F, _ = precision_recall_fscore_support(
        gt_labels, predictions, average="macro"
    )
    scores["Macro_P"] = macro_P
    scores["Macro_R"] = macro_R
    scores["Macro_F1"] = macro_F

    scores["Measurement_Classes_Macro_F1"] = 0.5 * (
        scores["MEASUREMENT"]["F"] + scores["QUAL_MEASUREMENT"]["F"]
    )

    return scores


def main() -> int:
    """
    Entry point.

    Returns:
        int: Exit code.
    """
    print_cmd_args(args)
    torch.manual_seed(args.seed)

    output_path: str = os.path.join(
        args.output_path, (f"cv_{args.cv}" if args.cv is not None else ".")
    )
    os.makedirs(output_path, exist_ok=True)

    use_tune: bool = args.cv is not None

    train_dataset: MeasurementDataset = MeasurementDataset(
        args.model_name, "train", args.cv, not args.include_non_meas_sents, args.subsample_rate
    )
    dev_dataset: MeasurementDataset = MeasurementDataset(
        args.model_name, "validation", filter_non_measurement_sents=not args.include_non_meas_sents
    )
    test_dataset: MeasurementDataset = MeasurementDataset(
        args.model_name, "test", filter_non_measurement_sents=not args.include_non_meas_sents
    )

    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    dev_dataloader: DataLoader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=args.batch_size)

    eval_dataloader: DataLoader = dev_dataloader

    if use_tune:
        tune_dataset: MeasurementDataset = MeasurementDataset(
            args.model_name, "tune", args.cv, not args.include_non_meas_sents
        )
        tune_dataloader: DataLoader = DataLoader(tune_dataset, batch_size=args.batch_size)
        eval_dataloader = tune_dataloader

    if args.use_random_classifier:
        train_dataset: MeasurementDataset = MeasurementDataset(
            args.model_name, "train", None, not args.include_non_meas_sents
        )
        random_classifier: RandomMeasurementClassifier = RandomMeasurementClassifier(train_dataset)
        P, R, F, S = precision_recall_fscore_support(
            test_dataset._data["labels"], random_classifier.predict(test_dataset)
        )
        # Since sklearn returns scores in sorted order, we sort our label set
        meas_labels.sort()
        for i, l in enumerate(meas_labels):
            logging.log(logging.INFO, f"Precision for label {l}: {P[i]}")
            logging.log(logging.INFO, f"Recall for label {l}: {R[i]}")
            logging.log(logging.INFO, f"F1 for label {l}: {F[i]}")
            logging.log(logging.INFO, f"Support for label {l}: {S[i]}")
        return

    classifier: MeasurementClassifier = MeasurementClassifier(
        args.model_name, (3 if args.include_non_meas_sents else 2), args.dropout_rate
    ).to(DEVICE)
    best_model: MeasurementClassifier = deepcopy(classifier)
    loss_function: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    optimizer: AdamW = AdamW(classifier.parameters(), lr=args.lr)
    if not args.disable_lr_decay:
        scheduler: LambdaLR = LambdaLR(
            optimizer=optimizer, lr_lambda=SqrtSchedule(len(train_dataloader))
        )

    best_score: float = 0.0
    early_stopping_counter: int = 0

    logging.log(logging.INFO, "Finished data loading. Starting training.")

    for epoch in range(args.num_epochs):
        logging.log(logging.INFO, f"Starting epoch {epoch + 1}/{args.num_epochs}")

        classifier = classifier.train()
        for batch in tqdm(train_dataloader):

            optimizer.zero_grad()

            if DEVICE != CPU:
                (
                    batch["tensor"]["input_ids"],
                    batch["tensor"]["attention_mask"],
                    batch["tensor"]["token_type_ids"],
                    batch["label"],
                ) = move_to_device(
                    DEVICE,
                    batch["tensor"]["input_ids"],
                    batch["tensor"]["attention_mask"],
                    batch["tensor"]["token_type_ids"],
                    batch["label"],
                )

            logits: torch.Tensor = classifier(
                batch["tensor"]["input_ids"],
                batch["tensor"]["attention_mask"],
                batch["tensor"]["token_type_ids"],
            )

            loss: torch.Tensor = loss_function(logits, batch["label"])
            loss.backward()
            optimizer.step()

            if not args.disable_lr_decay:
                scheduler.step()

            if DEVICE != CPU:
                (
                    batch["tensor"]["input_ids"],
                    batch["tensor"]["attention_mask"],
                    batch["tensor"]["token_type_ids"],
                    batch["label"],
                ) = move_to_device(
                    CPU,
                    batch["tensor"]["input_ids"],
                    batch["tensor"]["attention_mask"],
                    batch["tensor"]["token_type_ids"],
                    batch["label"],
                )

        logging.log(logging.INFO, "Starting evaluation.")
        scores = evaluate(classifier, eval_dataloader)
        logging.log(
            logging.INFO,
            f"Current Measurement Classes Macro F1: {scores['Measurement_Classes_Macro_F1']}",
        )
        if scores["Measurement_Classes_Macro_F1"] > best_score:
            best_model = deepcopy(classifier)
            best_score = scores["Measurement_Classes_Macro_F1"]
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter == 3:
            logging.log(
                logging.INFO,
                "Early stopping now since there hasn't been any improvement for the last 3 epochs.",
            )
            break

    final_dev_scores: dict = evaluate(best_model, dev_dataloader)
    final_test_scores: dict = evaluate(best_model, test_dataloader)
    with open(os.path.join(output_path, "scores_dev.pickle"), mode="wb") as f:
        pickle.dump(final_dev_scores, f)
    with open(os.path.join(output_path, "scores_test.pickle"), mode="wb") as f:
        pickle.dump(final_test_scores, f)
    return 0


if __name__ == "__main__":
    main()

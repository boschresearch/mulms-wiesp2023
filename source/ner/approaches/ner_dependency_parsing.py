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
This module contains the training pipeline for NER as dependency parsing.
"""

import logging
import os
import pickle
import sys
from argparse import ArgumentParser
from copy import deepcopy

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.constants.mulms_constants import CPU, CUDA, mulms_ne_dependency_labels
from source.ner.datasets.ner_dependency_dataset import NERDependencyDataset
from source.ner.dependency_graph.unfact_depgraph_parser import (
    UnfactorizedDependencyGraphParser,
)
from source.ner.evaluation.metrics import calculate_prec_recall_f1_for_dep_parsing
from source.ner.models.dependency_classifier import DependencyClassifier
from source.relation_extraction.model.embeddings.transformer_wrappers import BertWrapper
from source.relation_extraction.vocab import BasicVocab
from source.utils.helper_functions import print_cmd_args
from source.utils.lr_scheduler import SqrtSchedule

parser: ArgumentParser = ArgumentParser()
parser.add_argument("--model_name", type=str, help="Name of/Path to the pretrained model")
parser.add_argument(
    "--output_path", type=str, help="Storage path for fine-tuned model", default="."
)
parser.add_argument(
    "--store_model",
    action="store_true",
    help="Whether to store trained model (disable if storage space needs to be saved)",
)
parser.add_argument(
    "--disable_cuda", action="store_true", help="Disable CUDA in favour of CPU usage"
)
parser.add_argument("--seed", type=int, help="Random seed", default=23081861)
parser.add_argument("--batch_size", type=int, help="Batch size used during training", default=32)
parser.add_argument("--lr", type=float, help="Learning rate", default=4e-5)
parser.add_argument("--num_epochs", type=int, help="Number of training epochs", default=10)
parser.add_argument(
    "--disable_early_stopping",
    action="store_true",
    help="Whether to disable early stopping based on F1 score.",
)
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
    "--biaffine_hidden_size", type=int, help="Hidden size of biaffine scorer.", default=768
)

args = parser.parse_args()

DEVICE: str = CUDA if torch.cuda.is_available() and not args.disable_cuda else CPU
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def evaluate(model: UnfactorizedDependencyGraphParser, dataset: DataLoader) -> dict:
    """
    Evaluates a fine-tuned NER classifier.

    Args:
        model (UnfactorizedDependencyGraphParser): The fine-tuned Dep. Parser model.
        dataset (DataLoader): The data split to evaluate on.

    Returns:
        dict: Computed metrics (P, R, F1)
    """
    model = model.eval()
    predicted_dependencies: list[list[list[str]]] = []
    gt_dependencies: list[set[tuple]] = []
    with torch.no_grad():
        for batch in dataset:
            output = model(
                input_sents=batch["token_list"],
                targets=batch["input_batch"][2].to(DEVICE),
                mode="validation",
            )
            for p, g in zip(output[0], batch["instances"]):
                predicted_dependencies.append(p.dependencies)
                gt_dependencies.append(g._gold_labels)
    scores: dict = calculate_prec_recall_f1_for_dep_parsing(
        predicted_dependencies, gt_dependencies
    )
    return scores


def main():
    """
    Entry point.
    """
    print_cmd_args(args)

    use_tune_fold: bool = True
    if not args.cv:
        use_tune_fold = False

    torch.manual_seed(args.seed)

    output_path: str = os.path.join(args.output_path, f"cv_{args.cv}")

    os.makedirs(output_path, exist_ok=True)

    bert_wrapper: BertWrapper = BertWrapper(args.model_name).to(DEVICE)
    dependency_classifier: DependencyClassifier = DependencyClassifier(
        768,
        BasicVocab(mulms_ne_dependency_labels + ["O"], load_from_disk=False),
        "DeepBiaffineScorer",
        args.biaffine_hidden_size,
    ).to(DEVICE)
    dependency_graph_parser: UnfactorizedDependencyGraphParser = UnfactorizedDependencyGraphParser(
        bert_wrapper, dependency_classifier
    ).to(DEVICE)
    if use_tune_fold:
        train_ner_dataset: NERDependencyDataset = NERDependencyDataset.load_dataset(
            split="train", tune_id=args.cv
        )
        tune_ner_dataset: NERDependencyDataset = NERDependencyDataset.load_dataset(
            split="tune", tune_id=args.cv
        )
    else:
        train_ner_dataset: NERDependencyDataset = NERDependencyDataset.load_dataset(split="train")
    dev_ner_dataset: NERDependencyDataset = NERDependencyDataset.load_dataset(split="validation")
    test_ner_dataset: NERDependencyDataset = NERDependencyDataset.load_dataset(split="test")

    optimizer: AdamW = AdamW(dependency_graph_parser.parameters(), lr=args.lr, weight_decay=0)

    collate_fn = lambda x: NERDependencyDataset.collate_fn(  # noqa: E731
        x, dependency_graph_parser
    )
    epoch_eval_dataloader: DataLoader = None
    train_dataloader: DataLoader = DataLoader(
        dataset=train_ner_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True
    )
    dev_dataloader: DataLoader = DataLoader(
        dataset=dev_ner_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False
    )
    test_dataloader: DataLoader = DataLoader(
        dataset=test_ner_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False
    )

    if args.enable_lr_decay:
        scheduler: LambdaLR = LambdaLR(
            optimizer=optimizer, lr_lambda=SqrtSchedule(len(train_dataloader))
        )

    epoch_eval_dataloader: DataLoader = dev_dataloader
    if use_tune_fold:
        tune_dataloader: DataLoader = DataLoader(
            dataset=tune_ner_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )
        epoch_eval_dataloader = tune_dataloader

    best_f1: float = 0.0
    best_model: UnfactorizedDependencyGraphParser = deepcopy(dependency_graph_parser)
    early_stopping_counter: int = 0

    for epoch in range(args.num_epochs):
        logging.log(logging.INFO, f"Starting epoch {epoch + 1}/{args.num_epochs}")
        dependency_graph_parser = dependency_graph_parser.train()
        for batch in tqdm(train_dataloader):
            output = dependency_graph_parser(
                input_sents=batch["token_list"],
                targets=batch["input_batch"][2].to(DEVICE),
                mode="training",
            )
            output[1].backward()
            optimizer.step()
            optimizer.zero_grad()
        if args.enable_lr_decay:
            scheduler.step()

        logging.log(logging.INFO, "Evaluating")
        eval_scores: dict = evaluate(dependency_graph_parser, epoch_eval_dataloader)
        logging.log(logging.INFO, f"Current Micro F1: {eval_scores['Micro_F1']}")
        if eval_scores["Micro_F1"] > best_f1:
            best_f1 = eval_scores["Micro_F1"]
            early_stopping_counter = 0
            best_model = deepcopy(dependency_graph_parser)
            if args.store_model:
                torch.save(
                    best_model.state_dict(),
                    os.path.join(output_path, "ner_dependency_classifier.pt"),
                )
        else:
            early_stopping_counter += 1
        if (
            not args.disable_early_stopping and early_stopping_counter == 8
        ):  # Models tend to take a long initial phase before learning anything
            logging.log(logging.WARN, f"Early stopping after {epoch + 1} epochs")
            break

    final_dev_scores: dict = evaluate(best_model, dev_dataloader)
    final_test_scores: dict = evaluate(best_model, test_dataloader)
    with open(os.path.join(output_path, "scores_dev.pickle"), mode="wb") as f:
        pickle.dump(final_dev_scores, f)
    with open(os.path.join(output_path, "scores_test.pickle"), mode="wb") as f:
        pickle.dump(final_test_scores, f)
    return


if __name__ == "__main__":
    main()

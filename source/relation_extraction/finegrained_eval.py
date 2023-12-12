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
This module contains fine-grained evaluation methods for relation extraction on MuLMS.
Note: This script currently only supports MuLMS.
"""

import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.optim as optimizers_module
from model.relation_parser import MuLMSRelationParser
from torch.optim.lr_scheduler import LambdaLR

import source.relation_extraction.model.embeddings.transformer_wrappers as embeddings_module
from source.constants.mulms_constants import (
    MULMS_NE_VOCAB_PATH,
    MULMS_PATH,
    MULMS_RELATION_VOCAB_PATH,
)
from source.relation_extraction.data_handling.mulms_rel_dataloader import (
    get_relation_dataloader,
)
from source.relation_extraction.training.lr_schedules import *  # noqa: F401, F403
from source.relation_extraction.training.mulms_relsent_criterion import (
    MuLMSRelationSentenceValidationCriterion,
)
from source.relation_extraction.training.trainer import Trainer
from source.relation_extraction.vocab import BasicVocab


def init_model(model_config):
    transformer_model_class = getattr(embeddings_module, model_config["transformer_class"])
    transformer_model_path = model_config["transformer_model_path"]
    transformer_model_kwargs = model_config["transformer_model_kwargs"]
    transformer_model = transformer_model_class(
        model_path=transformer_model_path, **transformer_model_kwargs
    )

    ne_vocab = BasicVocab(vocab_filename=MULMS_NE_VOCAB_PATH)
    rel_vocab = BasicVocab(vocab_filename=MULMS_RELATION_VOCAB_PATH)

    model = MuLMSRelationParser(
        transformer_model, ne_vocab, rel_vocab, **model_config["model_kwargs"]
    )

    return model


def init_trainer(model, training_config, save_dir, tune_fold):
    # Data loaders
    train_data_loader = get_relation_dataloader(
        MULMS_PATH,
        "train",
        model,
        training_config["data_loaders"]["batch_size"],
        fold=tune_fold,
        reverse_fold=True,
    )
    tune_data_loader = get_relation_dataloader(
        MULMS_PATH,
        "train",
        model,
        training_config["data_loaders"]["batch_size"],
        fold=tune_fold,
        reverse_fold=False,
    )
    dev_data_loader = get_relation_dataloader(
        MULMS_PATH,
        "validation",
        model,
        training_config["data_loaders"]["batch_size"],
    )
    test_data_loader = get_relation_dataloader(
        MULMS_PATH,
        "test",
        model,
        training_config["data_loaders"]["batch_size"],
    )

    # Optimization stuff
    optimizer = getattr(optimizers_module, training_config["optimizer"]["type"])(
        model.parameters(), **training_config["optimizer"]["args"]
    )
    validation_criterion = MuLMSRelationSentenceValidationCriterion(
        training_config["validation_metric"]
    )
    lr_scheduler = LambdaLR(optimizer, eval(training_config["lr_schedule"]))

    return Trainer(
        model,
        training_config,
        optimizer,
        validation_criterion,
        train_data_loader,
        tune_data_loader,
        dev_data_loader,
        test_data_loader,
        save_dir,
        lr_scheduler=lr_scheduler,
    )


def get_single_model_metrics(model_path):
    model_path = Path(model_path)
    config_path = model_path / "config.json"

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    model = init_model(config["model"])
    trainer = init_trainer(model, config["training"], model_path, 1)

    return trainer.evaluate(return_raw_numbers=True)


if __name__ == "__main__":

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--input_path_one",
        type=str,
        help="Path to the results of the run with the first tune fold",
    )
    parser.add_argument(
        "--input_path_two",
        type=str,
        help="Path to the results of the run with the second tune fold",
    )
    parser.add_argument(
        "--input_path_three",
        type=str,
        help="Path to the results of the run with the third tune fold",
    )
    parser.add_argument(
        "--input_path_four",
        type=str,
        help="Path to the results of the run with the fourth tune fold",
    )
    parser.add_argument(
        "--input_path_five",
        type=str,
        help="Path to the results of the run with the fifth tune fold",
    )

    args = parser.parse_args()

    model_paths = [
        args.input_path_one,
        args.input_path_two,
        args.input_path_three,
        args.input_path_four,
        args.input_path_five,
    ]

    aggregated_model_metrics = defaultdict(list)

    for model_path in model_paths:
        dev_metrics, eval_metrics = get_single_model_metrics(model_path)
        for metric_name, metric_val in dev_metrics.items():
            aggregated_model_metrics[metric_name + "_dev"].append(metric_val)
        for metric_name, metric_val in eval_metrics.items():
            aggregated_model_metrics[metric_name + "_eval"].append(metric_val)

    avg_model_metrics = {
        metric: np.mean(metric_vals) for metric, metric_vals in aggregated_model_metrics.items()
    }
    std_model_metrics = {
        metric: np.std(metric_vals) for metric, metric_vals in aggregated_model_metrics.items()
    }

    print(avg_model_metrics["MICRO_f1_eval"])
    print(std_model_metrics["MICRO_f1_eval"])

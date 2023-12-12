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

import argparse
import json
from os import makedirs
from os.path import join
from pathlib import Path

import torch
import torch.optim as optimizers_module
from model.relation_parser import MuLMSRelationParser
from torch.optim.lr_scheduler import LambdaLR

import source.relation_extraction.model.embeddings.transformer_wrappers as embeddings_module
from source.constants.mulms_constants import (
    MSPT_NE_VOCAB_PATH,
    MSPT_RELATION_VOCAB_PATH,
    MULMS_NE_VOCAB_PATH,
    MULMS_PATH,
    MULMS_RELATION_VOCAB_PATH,
    SOFC_NE_VOCAB_PATH,
    SOFC_RELATION_VOCAB_PATH,
)
from source.relation_extraction.data_handling.mspt_relation_dataset import (
    get_mspt_relsent_dataloader,
)
from source.relation_extraction.data_handling.mulms_rel_dataloader import (
    get_relation_dataloader,
)
from source.relation_extraction.data_handling.sofc_relation_dataset import *  # noqa: F401 F403
from source.relation_extraction.data_handling.sofc_relation_dataset import (
    get_sofc_relsent_dataloader,
)
from source.relation_extraction.training.lr_schedules import (  # noqa: F401
    SqrtSchedule,
    TriangularSchedule,
    WarmRestartSchedule,
    WarmupSchedule,
)
from source.relation_extraction.training.mulms_relsent_criterion import (
    MuLMSRelationSentenceValidationCriterion,
)
from source.relation_extraction.training.trainer import Trainer
from source.relation_extraction.vocab import BasicVocab
from source.utils.helper_functions import get_executor_device

DEVICE: str = get_executor_device(disable_cuda=False)


def init_model(model_config: dict, training_config: dict):
    transformer_model_class = getattr(embeddings_module, model_config["transformer_class"])
    transformer_model_path = model_config["transformer_model_path"]
    transformer_model_kwargs = model_config["transformer_model_kwargs"]
    transformer_model = transformer_model_class(
        model_path=transformer_model_path, **transformer_model_kwargs
    )

    if training_config["target_dataset"] == "mulms":
        ne_vocab = BasicVocab(vocab_filename=MULMS_NE_VOCAB_PATH)
        rel_vocab = BasicVocab(vocab_filename=MULMS_RELATION_VOCAB_PATH)
    elif training_config["target_dataset"] == "sofc":
        ne_vocab = BasicVocab(vocab_filename=SOFC_NE_VOCAB_PATH)
        rel_vocab = BasicVocab(vocab_filename=SOFC_RELATION_VOCAB_PATH)
    elif training_config["target_dataset"] == "mspt":
        ne_vocab = BasicVocab(vocab_filename=MSPT_NE_VOCAB_PATH)
        rel_vocab = BasicVocab(vocab_filename=MSPT_RELATION_VOCAB_PATH)

    model = MuLMSRelationParser(
        transformer_model, ne_vocab, rel_vocab, **model_config["model_kwargs"]
    )

    return model


def init_trainer(model, training_config: dict, save_dir: str, tune_fold: int):
    # Data loaders
    if training_config["target_dataset"] == "mulms":
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

    elif training_config["target_dataset"] == "sofc":
        train_data_loader = get_sofc_relsent_dataloader(
            "train", model, training_config["data_loaders"]["batch_size"]
        )
        dev_data_loader = get_sofc_relsent_dataloader(
            "validation", model, training_config["data_loaders"]["batch_size"]
        )
        test_data_loader = get_sofc_relsent_dataloader(
            "test", model, 1
        )  # Hard-code batch size to 1 to deal with overly long sentences
    elif training_config["target_dataset"] == "mspt":
        train_data_loader = get_mspt_relsent_dataloader(
            "sfex-train", model, training_config["data_loaders"]["batch_size"]
        )
        dev_data_loader = get_mspt_relsent_dataloader(
            "sfex-dev", model, training_config["data_loaders"]["batch_size"]
        )
        test_data_loader = get_mspt_relsent_dataloader(
            "sfex-test", model, training_config["data_loaders"]["batch_size"]
        )
    else:
        assert False, "Wrong target dataset provided in config"

    # Optimization stuff
    optimizer = getattr(optimizers_module, training_config["optimizer"]["type"])(
        model.parameters(), **training_config["optimizer"]["args"]
    )
    validation_criterion = MuLMSRelationSentenceValidationCriterion(
        training_config["validation_metric"]
    )
    lr_scheduler = LambdaLR(optimizer, eval(training_config["lr_schedule"]))

    if training_config["target_dataset"] == "mulms":
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
    else:
        return Trainer(
            model,
            training_config,
            optimizer,
            validation_criterion,
            train_data_loader,
            dev_data_loader,
            dev_data_loader,
            test_data_loader,
            save_dir,
            lr_scheduler=lr_scheduler,
        )


def main(config: dict, tune_fold: int, save_dir: str) -> int:
    # Saving
    save_dir = Path(save_dir)
    run_name = config["run_name"]
    save_dir = join(save_dir, run_name, str(tune_fold))
    makedirs(save_dir)
    with open(join(save_dir, "config.json"), "w") as saved_config_file:
        json.dump(config, saved_config_file, indent=4, sort_keys=False)

    # Model
    model = init_model(config["model"], config["training"])
    model.save_config(save_dir, prefix="model")

    # Training and post-training evaluation
    trainer = init_trainer(model, config["training"], save_dir, tune_fold)
    trainer.train()
    trainer.evaluate()
    return 0


if __name__ == "__main__":
    torch.manual_seed(12345)

    argparser = argparse.ArgumentParser(description="MuLMS Relation Extraction")
    argparser.add_argument("--config", type=str, help="config file path (required)")
    argparser.add_argument(
        "--tune_fold",
        type=int,
        help="which fold of the train set to use for early stopping (1--5)",
    )
    argparser.add_argument("--save_dir", type=str, help="Output storage path")

    args = argparser.parse_args()
    with open(args.config, "r") as config_file:
        config: dict = json.load(config_file)

    main(config, args.tune_fold, args.save_dir)

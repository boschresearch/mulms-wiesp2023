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
from source.data_handling.mspt_dataset import *  # noqa: F403
from source.relation_extraction.data_handling.mspt_relation_dataset import (
    get_mspt_relsent_dataloader,
)
from source.relation_extraction.data_handling.mulms_rel_dataloader import (
    get_relation_dataloader,
)
from source.relation_extraction.data_handling.multitask_dataloader import (
    MultitaskDataloader,
)
from source.relation_extraction.data_handling.sofc_relation_dataset import *  # noqa: F403
from source.relation_extraction.model.multitask_relation_parser import (
    MultitaskRelationParser,
)
from source.relation_extraction.model.relation_parser import MuLMSRelationParser
from source.relation_extraction.training.lr_schedules import *  # noqa: F403
from source.relation_extraction.training.mulms_relsent_criterion import (
    MuLMSRelationSentenceValidationCriterion,
)
from source.relation_extraction.training.trainer import Trainer
from source.relation_extraction.vocab import BasicVocab

dataset_to_idx: dict = {"mulms": -1, "sofc": -1, "mspt": -1}


def init_multitask_model(model_config: dict):
    assert isinstance(model_config, list)

    relation_parsers = list()
    for i, rel_parser_config in enumerate(model_config):
        rel_parser = init_submodel(rel_parser_config)
        relation_parsers.append(rel_parser)
        dataset_to_idx[rel_parser_config["dataset"]] = i

    return MultitaskRelationParser(relation_parsers)


def init_submodel(model_config: dict):
    transformer_model_class = getattr(embeddings_module, model_config["transformer_class"])
    transformer_model_path = model_config["transformer_model_path"]
    transformer_model_kwargs = model_config["transformer_model_kwargs"]
    transformer_model = transformer_model_class(
        model_path=transformer_model_path, **transformer_model_kwargs
    )

    if model_config["dataset"] == "mulms":
        mulms_ne_vocab = BasicVocab(vocab_filename=MULMS_NE_VOCAB_PATH)
        mulms_rel_vocab = BasicVocab(vocab_filename=MULMS_RELATION_VOCAB_PATH)
        model = MuLMSRelationParser(
            transformer_model, mulms_ne_vocab, mulms_rel_vocab, **model_config["model_kwargs"]
        )
    elif model_config["dataset"] == "sofc":
        sofc_ne_vocab = BasicVocab(vocab_filename=SOFC_NE_VOCAB_PATH)
        sofc_rel_vocab = BasicVocab(vocab_filename=SOFC_RELATION_VOCAB_PATH)
        model = MuLMSRelationParser(
            transformer_model, sofc_ne_vocab, sofc_rel_vocab, **model_config["model_kwargs"]
        )
    elif model_config["dataset"] == "mspt":
        mspt_ne_vocab = BasicVocab(vocab_filename=MSPT_NE_VOCAB_PATH)
        mspt_rel_vocab = BasicVocab(vocab_filename=MSPT_RELATION_VOCAB_PATH)
        model = MuLMSRelationParser(
            transformer_model, mspt_ne_vocab, mspt_rel_vocab, **model_config["model_kwargs"]
        )

    return model


def init_trainer(model, training_config, save_dir, tune_fold):
    # NOTE: The models provided to the get_dataloader functions use hard-coded indices (0, 1, 2)
    # So the first submodel must always be the one for MuLMS, the second the one for SOFC, the third one for MSPT

    # MuLMS data loaders
    if dataset_to_idx["mulms"] != -1:
        mulms_train_data_loader = get_relation_dataloader(
            MULMS_PATH,
            "train",
            model.relation_parsers[dataset_to_idx["mulms"]],
            training_config["data_loaders"]["batch_size"],
            fold=tune_fold,
            reverse_fold=True,
        )
        mulms_tune_data_loader = get_relation_dataloader(
            MULMS_PATH,
            "train",
            model.relation_parsers[dataset_to_idx["mulms"]],
            training_config["data_loaders"]["batch_size"],
            fold=tune_fold,
            reverse_fold=False,
        )
        mulms_dev_data_loader = get_relation_dataloader(
            MULMS_PATH,
            "validation",
            model.relation_parsers[dataset_to_idx["mulms"]],
            training_config["data_loaders"]["batch_size"],
        )
        mulms_test_data_loader = get_relation_dataloader(
            MULMS_PATH,
            "test",
            model.relation_parsers[dataset_to_idx["mulms"]],
            training_config["data_loaders"]["batch_size"],
        )

    # SOFC data loader
    if dataset_to_idx["sofc"] != -1:
        sofc_train_data_loader = get_sofc_relsent_dataloader(  # noqa: F405
            "train",
            model.relation_parsers[dataset_to_idx["sofc"]],
            training_config["data_loaders"]["batch_size"],
        )
        sofc_dev_data_loader = get_sofc_relsent_dataloader(  # noqa: F405
            "validation",
            model.relation_parsers[dataset_to_idx["sofc"]],
            training_config["data_loaders"]["batch_size"],
        )
        sofc_test_data_loader = get_sofc_relsent_dataloader(  # noqa: F405
            "test", model.relation_parsers[dataset_to_idx["sofc"]], 1
        )  # Hard-code batch size to 1 to deal with overly long sentences

    # MSPT data loader
    if dataset_to_idx["mspt"] != -1:
        mspt_train_data_loader = get_mspt_relsent_dataloader(
            "sfex-train",
            model.relation_parsers[dataset_to_idx["mspt"]],
            training_config["data_loaders"]["batch_size"],
        )
        mspt_dev_data_loader = get_mspt_relsent_dataloader(
            "sfex-dev",
            model.relation_parsers[dataset_to_idx["mspt"]],
            training_config["data_loaders"]["batch_size"],
        )
        mspt_test_data_loader = get_mspt_relsent_dataloader(
            "sfex-test",
            model.relation_parsers[dataset_to_idx["mspt"]],
            training_config["data_loaders"]["batch_size"],
        )

    # Joint dataloader based on the current config
    if (
        training_config["target_dataset"] == "mulms"
        and training_config["additional_datasets"] == "both"
    ):
        joint_train_data_loader = MultitaskDataloader(
            mulms_train_data_loader, sofc_train_data_loader, mspt_train_data_loader
        )
    elif (
        training_config["target_dataset"] == "mulms"
        and training_config["additional_datasets"] == "sofc"
    ):
        joint_train_data_loader = MultitaskDataloader(
            mulms_train_data_loader, sofc_train_data_loader
        )
    elif (
        training_config["target_dataset"] == "mulms"
        and training_config["additional_datasets"] == "mspt"
    ):
        joint_train_data_loader = MultitaskDataloader(
            mulms_train_data_loader, mspt_train_data_loader
        )
    elif (
        training_config["target_dataset"] == "sofc"
        and training_config["additional_datasets"] == "mulms"
    ):
        joint_train_data_loader = MultitaskDataloader(
            sofc_train_data_loader, mulms_train_data_loader
        )
    elif (
        training_config["target_dataset"] == "sofc"
        and training_config["additional_datasets"] == "mspt"
    ):
        joint_train_data_loader = MultitaskDataloader(
            sofc_train_data_loader, mspt_train_data_loader
        )
    elif (
        training_config["target_dataset"] == "mspt"
        and training_config["additional_datasets"] == "mulms"
    ):
        joint_train_data_loader = MultitaskDataloader(
            mspt_train_data_loader, mulms_train_data_loader
        )
    elif (
        training_config["target_dataset"] == "mspt"
        and training_config["additional_datasets"] == "sofc"
    ):
        joint_train_data_loader = MultitaskDataloader(
            mspt_train_data_loader, sofc_train_data_loader
        )

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
            joint_train_data_loader,
            mulms_tune_data_loader,
            mulms_dev_data_loader,
            mulms_test_data_loader,
            save_dir,
            lr_scheduler=lr_scheduler,
        )
    elif training_config["target_dataset"] == "sofc":
        return Trainer(
            model,
            training_config,
            optimizer,
            validation_criterion,
            joint_train_data_loader,
            sofc_dev_data_loader,
            sofc_dev_data_loader,
            sofc_test_data_loader,
            save_dir,
            lr_scheduler=lr_scheduler,
        )
    elif training_config["target_dataset"] == "mspt":
        return Trainer(
            model,
            training_config,
            optimizer,
            validation_criterion,
            joint_train_data_loader,
            mspt_dev_data_loader,
            mspt_dev_data_loader,
            mspt_test_data_loader,
            save_dir,
            lr_scheduler=lr_scheduler,
        )


def main(config: dict, tune_fold: int, save_dir: str):
    # Saving
    save_dir = Path(save_dir)
    run_name = config["run_name"]
    save_dir = join(save_dir, run_name, str(tune_fold))
    makedirs(save_dir)
    with open(join(save_dir, "config.json"), "w") as saved_config_file:
        json.dump(config, saved_config_file, indent=4, sort_keys=False)

    # Model
    model = init_multitask_model(config["model"])
    model.save_config(save_dir, prefix="model")

    # Training and post-training evaluation
    trainer = init_trainer(model, config["training"], save_dir, tune_fold)

    trainer.train()
    trainer.evaluate()


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
        config = json.load(config_file)

    main(config, args.tune_fold, args.save_dir)

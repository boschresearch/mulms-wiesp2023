#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

"""
This module contains the NER as dependency parsing dataset class for the MuLMS corpus.
"""

from math import inf

import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset

from source.constants.mulms_constants import (
    MULMS_DATASET_READER_PATH,
    MULMS_FILES,
    MULMS_PATH,
)
from source.ner.dependency_graph.unfact_depgraph import UnfactorizedDependencyGraph
from source.ner.dependency_graph.unfact_depgraph_parser import (
    UnfactorizedDependencyGraphParser,
)


class NERDependencyDataset(TorchDataset):
    """
    An object of this class represents a (map-style) dataset of instances read from a CoNLL-like format.
    The individual objects contained within the dataset can be any output structure.
    """

    @staticmethod
    def collate_fn(data: list, model: UnfactorizedDependencyGraphParser) -> dict:
        """
        Returns a dict containing an instance of the dataset, including its factor graph.

        Args:
            data (list): NER data
            model (UnfactorizedDependencyGraphParser): Factor graph

        Returns:
            dict: Dict containing ID, tokens and the dependency graphs
        """
        output: dict = {}
        output["ID"] = [None] * len(data)
        output["token_list"] = [None] * len(data)
        output["instances"] = [None] * len(data)
        for i, d in enumerate(data):
            output["ID"][i] = d["ID"]
            output["token_list"][i] = d["token_list"]
            output["instances"][i] = d["instance"]
        output["input_batch"] = UnfactorizedDependencyGraph.batchify(output["instances"], model)
        return output

    def __init__(self) -> None:
        """
        Initializes the dataset by creating lists that hold the parser graph structures.
        """
        self._instances: list[UnfactorizedDependencyGraph] = []
        self._tokens: list[list[str]] = []

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self._instances)

    def __getitem__(self, index: int) -> dict:
        """
        Returns a specific sample of the dataset.

        Args:
            index (int): Index to the instance

        Returns:
            dict: Dict containing ID, list of tokens and dependency graph instance
        """
        return {"ID": index, "token_list": self._tokens[index], "instance": self._instances[index]}

    def append_instance(self, instance: UnfactorizedDependencyGraph) -> None:
        """
        Append one instance to the dataset.

        Args:
            instance (UnfactorizedDependencyGraph): Instance object to add to the dataset.
        """
        if instance is not None:
            self._instances.append(instance)

    def append_token_list(self, tokens: list[str]) -> None:
        """
        Appends list of tokens to internal list of lists.

        Args:
            tokens (list[str]): List of tokens.
        """
        if tokens is not None:
            self._tokens.append(tokens)

    @staticmethod
    def load_dataset(
        output_structure=UnfactorizedDependencyGraph,
        max_sent_len: int = inf,
        split: str = "train",
        tune_id: int = None,
    ):
        """
        Read in a dataset from a corpus file in CoNLL format.

        Args:
            output_structure (optional): Which kind of output structure to fill the dataset with. Defaults to UnfactorizedDependencyGraph.
            max_sent_len (int, optional): he maximum length of any given sentence. Sentences with a greater length are ignored. Defaults to inf.
            split (str, optional): Which split to load. Must be train/tune/validation/test. Defaults to train.
            tune_id (int, optional): Which of the 5 train splits to use as tune fold. If set and split==train, all samples with this ID will be filtered. Defaults to None.

        Returns:
            NERDependencyDataset: A NERDependencyDataset object containing the sentences in the input corpus file, with the specified annotation
            layers.
        """
        assert split in ["train", "tune", "validation", "test"], "Invalid split specified."
        ner_dataset = NERDependencyDataset()

        ner_dep_hf_dataset: Dataset = load_dataset(
            MULMS_DATASET_READER_PATH.__str__(),
            data_dir=MULMS_PATH.__str__(),
            data_files=MULMS_FILES,
            name="NER_Dependencies",
            split="train" if split == "tune" else split,
        )

        ids: np.ndarray = np.array(ner_dep_hf_dataset["ID"])
        token_ids: np.ndarray = np.array(ner_dep_hf_dataset["token_id"])
        token_texts: np.ndarray = np.array(ner_dep_hf_dataset["token_text"])
        label: np.ndarray = np.array(ner_dep_hf_dataset["NE_Dependencies"])
        splits: np.ndarray = np.array(ner_dep_hf_dataset["data_split"])
        for id in set(ner_dep_hf_dataset["ID"]):
            indices: np.ndarray = ids == id
            if len(set(splits[indices])) > 1:
                raise
            if (splits[indices][0] == f"train{tune_id}" and split == "train") or (
                splits[indices][0] != f"train{tune_id}" and split == "tune"
            ):
                continue
            t_ids: np.ndarray = token_ids[indices]
            t_texts: np.ndarray = token_texts[indices]
            t_label: np.ndarray = label[indices]
            reader_result = output_structure.load_from_dataset([t_ids, t_texts, t_label])
            instance = reader_result
            if len(instance) <= max_sent_len:
                ner_dataset.append_instance(instance)
                ner_dataset.append_token_list(t_texts.tolist())

        return ner_dataset

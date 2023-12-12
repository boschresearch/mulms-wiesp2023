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
This module contains the dataset class for the measurement detection task.
"""

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from source.constants.mulms_constants import (
    MULMS_DATASET_READER_PATH,
    MULMS_FILES,
    MULMS_PATH,
    meas_label2id,
)


class MeasurementDataset(Dataset):
    """
    The measurement classification dataset that prepares MuLMS for the task.
    """

    def __init__(
        self,
        tokenizer_model_name: str,
        split: str,
        tune_id: int = None,
        filter_non_measurement_sents: bool = True,
        subsample_rate: float = 1.0,
        seed: int = 23081861,
    ) -> None:
        """
        Initilaizes the dataset.

        Args:
            tokenizer_model_name (str): Name or path of the BERT-based tokenizer.
            split (str): Desired split; must be one of [train, tune, dev, test]
            tune_id (int, optional): ID of the tune split. Defaults to None.
            filter_non_measurement_sents (bool, optional): Whether to filter out non-measurement sentences. Defaults to True.
            subsample_rate (float, optional): Rate of subsampling. Defaults to 1.0.
            seed (int, optional): Random seed for random subsampling. Defaults to 23081861.
        """
        super().__init__()
        assert split in ["train", "tune", "validation", "test"], "Invalid split specified."
        assert (
            subsample_rate <= 1.0 and subsample_rate >= 0.0
        ), "Subsample rate must be within [0.0, 1.0]."
        self._split: str = split
        self._tune_id: int = tune_id
        self._filter_non_measurement_sents: bool = filter_non_measurement_sents
        self._seed = seed
        self._data: dict = {
            "ids": None,
            "sentences": [],
            "tokens": [],
            "labels": [],
            "label_ids": [],
        }
        self._tensor_encoded_data: dict = None
        self._tensor_encoded_labels: torch.Tensor = None
        self._load_data(tokenizer_model_name, subsample_rate)

    def __getitem__(self, index: int) -> dict:
        """
        Returns a sample of the dataset.

        Args:
            index (int): Index to the specific sample.

        Returns:
            dict: Dict containing sample ID, BERT tensors and label.
        """
        return {
            "id": self._data["ids"][index],
            "tensor": {
                "input_ids": self._tensor_encoded_data["input_ids"][index],
                "attention_mask": self._tensor_encoded_data["attention_mask"][index],
                "token_type_ids": self._tensor_encoded_data["token_type_ids"][index],
            },
            "label": self._tensor_encoded_labels[index],
        }

    def __len__(self) -> int:
        """
        Returns length of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self._tensor_encoded_labels)

    def _load_data(self, tokenizer_model_name: str, subsample_rate: float = 1.0):
        """
        Loads data from the disk and prepares it for the BERT-based model.

        Args:
            tokenizer_model_name (str): Name or path of BERT-based tokenizer.
            subsample_rate (float, optional): Subsampling rate. Defaults to 1.0.
        """
        dataset: HFDataset = load_dataset(
            MULMS_DATASET_READER_PATH.__str__(),
            data_dir=MULMS_PATH.__str__(),
            data_files=MULMS_FILES,
            name="MuLMS_Corpus",
            split=("train" if self._split == "tune" else self._split),
        )
        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_model_name)
        for sent, tokens, label, data_split in zip(
            dataset["sentence"],
            dataset["tokens"],
            dataset["Measurement_label"],
            dataset["data_split"],
        ):
            if self._tune_id is not None and self._split == "train":
                if f"train{self._tune_id}" == data_split:
                    continue
            elif self._tune_id is not None and self._split == "tune":
                if f"train{self._tune_id}" != data_split:
                    continue
            if self._filter_non_measurement_sents:
                if label == "O":
                    continue
            self._data["sentences"].append(sent)
            self._data["tokens"].append(tokens)
            self._data["labels"].append(label)
            self._data["label_ids"].append(meas_label2id[label])

        if subsample_rate < 1.0 and not self._filter_non_measurement_sents:
            non_measurement_label: str = meas_label2id["O"]
            non_measurement_indices: np.ndarray = (
                np.array(self._data["label_ids"]) == non_measurement_label
            ).nonzero()
            random_generator = np.random.default_rng(seed=self._seed)
            removable_indices: np.ndarray = random_generator.choice(
                non_measurement_indices[0],
                int(len(non_measurement_indices[0]) * (1 - subsample_rate)),
                replace=False,
            )
            self._data["sentences"] = list(np.delete(self._data["sentences"], removable_indices))
            self._data["tokens"] = list(np.delete(self._data["tokens"], removable_indices))
            self._data["labels"] = list(np.delete(self._data["labels"], removable_indices))
            self._data["label_ids"] = list(np.delete(self._data["label_ids"], removable_indices))

        self._tensor_encoded_data = tokenizer.batch_encode_plus(
            self._data["tokens"], is_split_into_words=True, return_tensors="pt", padding=True
        )
        self._tensor_encoded_labels = torch.tensor(self._data["label_ids"])
        self._data["ids"] = list(range(1, len(self._tensor_encoded_labels) + 1))
        return

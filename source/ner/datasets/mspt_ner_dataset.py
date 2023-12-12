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
This module contains the NER dataset class for the MSPT corpus used in the multi-task experiments.
"""

from typing import Tuple

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import BertTokenizer

from source.constants.mulms_constants import (
    mspt_bilou_id2ne_label,
    mspt_bilou_ne_label2id,
    mspt_ne_labels,
)
from source.data_handling.mspt_dataset import MSPT_Dataset


class MSPT_NER_Dataset(TorchDataset):
    """
    NER dataset class for the MSPT corpus.
    """

    def __init__(self, split: str, tokenizer_model_name: str) -> None:
        """
        Initializes the dataset by reading the files from disk and preparing BERT-based tensors.

        Args:
            split (str): Desired split. Must be one of [ner-train, ner-dev, ner-test]
            tokenizer_model_name (str): Path or name of the underlying BERT-based model.
        """
        assert split in [
            "ner-train",
            "ner-dev",
            "ner-test",
        ], "Invalid split provided. Split must be one of: ner-train, ner-dev, ner-test"
        self.name: str = "MSPT_NER"
        self.ne_labels: list[str] = mspt_ne_labels
        self.bilou_ne_label2id: dict = mspt_bilou_ne_label2id
        self.bilou_id2ne_label: dict = mspt_bilou_id2ne_label
        self._split = split
        self._ner_data: dict = {
            "id": [],
            "sentences": [],
            "tokens": [],
            "ne_labels": [],
            "ne_labels_bilou": [],
            "ne_labels_bilou_string": [],
            "crf_mask": [],
            "tensor_encoded_input": None,
        }  # This dict contains the data unrolled s.t. it can be iterated over
        self._ne_annot_list: list[set[tuple]] = []
        self._load_mspt_ner_dataset(tokenizer_model_name=tokenizer_model_name)

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self._ner_data["sentences"])

    def __getitem__(self, index: int) -> dict:
        """
        Returns a specific sample of the dataset.

        Args:
            index (int): Index to the sample

        Returns:
            dict: Dict containing ID, raw text, BERT tensors, (sorted) CRF mask, label IDs and text-based labels.
        """
        return {
            "id": self._ner_data["id"][index],
            "text": self._ner_data["sentences"][index],
            "input_ids": self._ner_data["tensor_encoded_input"]["input_ids"][index],
            "attention_mask": self._ner_data["tensor_encoded_input"]["attention_mask"][index],
            "token_type_ids": self._ner_data["tensor_encoded_input"]["token_type_ids"][index],
            "crf_mask": self._ner_data["crf_mask"][index],
            "sorted_crf_mask": self._ner_data["sorted_crf_mask"][index],
            "label_ids": self._ner_data["ne_label_ids"][index],
            "label": str(
                self._ner_data["ne_labels"][index]
            ),  # Hack to make it equal sized for batch generation
        }

    def get_token_list(self) -> list[list[str]]:
        """
        Returns a list of lists containing the tokens.

        Returns:
            list[list[str]]: Nested token lists
        """
        return self._ner_data["tokens"]

    def get_gold_labels(self) -> Tuple:
        """
        Retrieve all gold label annotations as list to save time.
        Only useful when compared to non-shuffled batches!

        Returns:
            Tuple: Contains lists of different label formats (IDs, strings, tuples with start and end)
        """
        return (
            self._ner_data["ne_labels_bilou"],
            self._ner_data["ne_labels_bilou_string"],
            self._ne_annot_list,
        )

    def _load_mspt_ner_dataset(self, tokenizer_model_name: str) -> None:
        """
        Loads and prepares MSPT dataset from disk.

        Args:
            tokenizer_model_name (str): Path or name of the underlying BERT-based model.
        """
        mspt_dataset: MSPT_Dataset = MSPT_Dataset(self._split)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
        B: str = "B-{0}"
        I: str = "I-{0}"
        L: str = "L-{0}"
        U: str = "U-{0}"
        O: str = "O"
        id: int = 0
        for doc_id, named_entity_lookup in mspt_dataset._named_entities.items():
            for sent_id, named_entities in named_entity_lookup.items():
                self._ner_data["id"].append(id)
                self._ner_data["sentences"].append(
                    mspt_dataset._sentences_as_string[doc_id][sent_id]
                )
                self._ner_data["tokens"].append(mspt_dataset._token_list[doc_id][sent_id])
                self._ner_data["ne_labels"].append(named_entities)
                self._ner_data["ne_labels_bilou_string"].append(
                    [O] * len(mspt_dataset._token_list[doc_id][sent_id])
                )

                self._ne_annot_list.append(set())

                for ent in named_entities:
                    label: str = ent.label
                    # Unit length case
                    if ent.begin_token == ent.end_token:
                        self._ner_data["ne_labels_bilou_string"][-1][ent.begin_token] = U.format(
                            label
                        )
                    # B-I-L required
                    else:
                        self._ner_data["ne_labels_bilou_string"][-1][ent.begin_token] = B.format(
                            label
                        )
                        for j in range(ent.begin_token + 1, ent.end_token):
                            self._ner_data["ne_labels_bilou_string"][-1][j] = I.format(label)
                        self._ner_data["ne_labels_bilou_string"][-1][ent.end_token] = L.format(
                            label
                        )

                    self._ne_annot_list[-1].add(tuple([label, ent.begin_token, ent.end_token]))

                self._ner_data["ne_labels_bilou"].append(
                    [
                        mspt_bilou_ne_label2id[label]
                        for label in self._ner_data["ne_labels_bilou_string"][-1]
                    ]
                )

                crf_mask: list[int] = [0]  # [CLS]

                for t in mspt_dataset._token_list[doc_id][sent_id]:
                    ids: list[int] = tokenizer.encode(t)
                    if len(ids) >= 3:  # Skip if white space has been tokenized
                        crf_mask.extend(
                            [1] + [0] * (len(ids) - 3)
                        )  # WordPiece tokens - (actual token + [CLS] + [SEP])

                crf_mask.append(0)  # [SEP]
                self._ner_data["crf_mask"].append(crf_mask)

                id += 1

        self._ner_data["tensor_encoded_input"] = tokenizer.batch_encode_plus(
            self._ner_data["tokens"], is_split_into_words=True, return_tensors="pt", padding=True
        )
        pad_size: int = self._ner_data["tensor_encoded_input"]["input_ids"].shape[1]
        self._ner_data["crf_mask"] = torch.tensor(
            [c + [0] * (pad_size - len(c)) for c in self._ner_data["crf_mask"]], dtype=torch.uint8
        )  # Padded CRF masks
        self._ner_data["sorted_crf_mask"] = torch.sort(
            self._ner_data["crf_mask"], dim=1, descending=True, stable=True
        )[
            0
        ]  # Used for cutting output of CRF layer
        self._ner_data["ne_label_ids"] = torch.tensor(
            [
                label + [mspt_bilou_ne_label2id["x"]] * (pad_size - len(label))
                for label in self._ner_data["ne_labels_bilou"]
            ]
        )

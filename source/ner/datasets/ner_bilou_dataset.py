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
This module contains the BILOU NER dataset class for the MuLMS corpus.
"""

from typing import Tuple

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import BertTokenizer, BertTokenizerFast

from source.constants.mulms_constants import (
    MULMS_DATASET_READER_PATH,
    MULMS_FILES,
    MULMS_PATH,
    mulms_ne_labels,
)
from source.constants.nested_bilou_tags import (
    mulms_bilou_id2ne_label,
    mulms_bilou_ne_label2id,
)


class MULMS_NER_BILOU_Dataset(TorchDataset):
    """
    MuLMS NER dataset class that uses the BILOU tagging scheme, which uses the B- (begin), I- (inside), L- (last), O- (outside), and U- (unit) markers for token-wise labelling.
    """

    def __init__(self, split: str, tokenizer_model_name: str, tune_id: int = None) -> None:
        """
        Initializes the MuLMS NER BILOU dataset.

        Args:
            split (str): Desired split. Must be one of [train, tune, validation, test].
            tokenizer_model_name (str): Name or path of BERT-based tokenizer.
            tune_id (int, optional): If split == tune, this corresponds to one of the 5 possible tune splits. Defaults to None.
        """
        super().__init__()
        _tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
        _fast_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_model_name)
        assert split in [
            "train",
            "tune",
            "validation",
            "test",
        ], "Split must be train/validation/test."
        self._split = split
        self.name: str = "NER_BILOU"
        self.ne_labels: list[str] = mulms_ne_labels
        self.bilou_ne_label2id: dict = mulms_bilou_ne_label2id
        self.bilou_id2ne_label: dict = mulms_bilou_id2ne_label
        self._ner_data: dict = {}
        self._tag_list: list[list[str]] = None
        self._ne_annot_list: list[set[tuple]] = []
        self._token_list: list[list[str]] = None
        self._indices: list[int] = []  # Used for distinguishing train and tune splits
        self._load_ner_bilou_dataset(
            tokenizer=_tokenizer, fast_tokenizer=_fast_tokenizer, tune_id=tune_id
        )
        self._tag_id_list: list[list[int]] = [
            [mulms_bilou_ne_label2id[t] for t in tl] for tl in self._tag_list
        ]
        del _tokenizer
        del _fast_tokenizer

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self._indices)

    def __getitem__(self, index: int) -> dict:
        """
        Returns a specific sample of the dataset.

        Args:
            index (int): Index to the sample

        Returns:
            dict: Dict containing ID, document ID, plain text, BERT tensors, NE labels, IDs of labels, CRF mask and sorted CRF mask.
        """
        idx: int = self._indices[index]
        return {
            "id": self._ner_data["ID"][idx],
            "doc_id": self._ner_data["doc_id"][idx],
            "text": self._ner_data["text"][idx],
            "input_ids": self._ner_data["input_ids"][idx],
            "attention_mask": self._ner_data["attention_mask"][idx],
            "token_type_ids": self._ner_data["token_type_ids"][idx],
            "label": str(
                self._ner_data["label"][idx]
            ),  # Hack to make it equal sized for batch generation
            "label_ids": self._ner_data["label_ids"][idx],
            "crf_mask": self._ner_data["crf_mask"][idx],
            "sorted_crf_mask": self._ner_data["sorted_crf_mask"][idx],
        }

    def get_token_list(self) -> list[list[str]]:
        """
        Returns a nested list of token lists.

        Returns:
            list[list[str]]: Batch of token lists.
        """
        return self._token_list

    def get_gold_labels(self) -> Tuple:
        """
        Retrieve all gold label annotations as list to save time.
        Only useful when compared to non-shuffled batches!

        Returns:
            Tuple: Contains label IDs, labels and span-based NE annotations
        """
        return self._tag_id_list, self._tag_list, self._ne_annot_list

    def _load_ner_bilou_dataset(
        self, tokenizer: BertTokenizer, fast_tokenizer: BertTokenizerFast, tune_id=None
    ) -> None:
        """
        Loads and prepared the MuLMS NER dataset.

        Args:
            tokenizer (BertTokenizer): Path or name of BERT-based tokenizer.
            fast_tokenizer (BertTokenizerFast): Fast variant of the tokenizer. Used for creating the large batch after having computed sub-word tokens that are required for CRF masks.
            tune_id (_type_, optional): If split == tune, this corresponds to one of the 5 possible tune splits. Defaults to None.
        """
        ner_hf_dataset: HFDataset = load_dataset(
            MULMS_DATASET_READER_PATH.__str__(),
            data_dir=MULMS_PATH.__str__(),
            data_files=MULMS_FILES,
            name="MuLMS_Corpus",
            split=("train" if self._split == "tune" else self._split),
        )
        crf_masks: list[list[int]] = []
        labels: list[list[str]] = []
        label_ids: list[list[int]] = []
        self._token_list: list[list[str]] = ner_hf_dataset[
            "tokens"
        ]  # Required since iterating directly is extremely slow
        labels_list: list[list[str]] = ner_hf_dataset["NER_labels_BILOU"]
        for i in range(len(self._token_list)):
            labels.append(labels_list[i])
            label_ids.append([])
            crf_masks.append([0])
            for t, l in zip(self._token_list[i], labels_list[i]):
                ids: list[int] = tokenizer.encode(t)
                if len(ids) >= 3:  # Skip if white space has been tokenized
                    crf_masks[-1].extend(
                        [1] + [0] * (len(ids) - 3)
                    )  # WordPiece tokens - (actual token + [CLS] + [SEP])
                    label_ids[-1].extend([mulms_bilou_ne_label2id[l]])
                else:
                    pass
            crf_masks[-1].append(0)
        encoded_input = fast_tokenizer(
            ner_hf_dataset["tokens"], is_split_into_words=True, return_tensors="pt", padding=True
        )
        self._ner_data["ID"] = list(range(1, len(self._token_list) + 1))
        self._ner_data["doc_id"] = ner_hf_dataset["doc_id"]
        self._ner_data["text"] = ner_hf_dataset["sentence"]
        self._ner_data["tokens"] = self._token_list
        self._ner_data["input_ids"] = encoded_input.data["input_ids"]
        self._ner_data["attention_mask"] = encoded_input.data["attention_mask"]
        self._ner_data["token_type_ids"] = encoded_input.data["token_type_ids"]
        self._ner_data["label"] = labels
        pad_size: int = self._ner_data["input_ids"].shape[1]
        self._ner_data["crf_mask"] = torch.tensor(
            [c + [0] * (pad_size - len(c)) for c in crf_masks], dtype=torch.uint8
        )  # Padded CRF masks
        self._ner_data["sorted_crf_mask"] = torch.sort(
            self._ner_data["crf_mask"], dim=1, descending=True, stable=True
        )[
            0
        ]  # Used for cutting output of CRF layer
        self._ner_data["label_ids"] = torch.tensor(
            [nl + [mulms_bilou_ne_label2id["x"]] * (pad_size - len(nl)) for nl in label_ids]
        )
        self._tag_list = ner_hf_dataset["NER_labels_BILOU"]
        for ne_list in ner_hf_dataset["NER_labels"]:
            self._ne_annot_list.append(set())
            for i in range(len(ne_list["text"])):
                self._ne_annot_list[-1].add(
                    tuple(
                        [
                            ne_list["value"][i],
                            ne_list["tokenIndices"][i][0],
                            ne_list["tokenIndices"][i][1],
                        ]
                    )
                )
        if tune_id is not None:
            if self._split == "train":
                self._indices = [
                    j for j, s in enumerate(ner_hf_dataset["data_split"]) if s != f"train{tune_id}"
                ]
            elif self._split == "tune":
                self._indices = [
                    j for j, s in enumerate(ner_hf_dataset["data_split"]) if s == f"train{tune_id}"
                ]
            else:
                self._indices = list(range(len(ner_hf_dataset["data_split"])))
        else:
            self._indices = list(range(len(ner_hf_dataset["data_split"])))

        if len(self._indices) < len(self._token_list):
            self._token_list = [tl for j, tl in enumerate(self._token_list) if j in self._indices]
            self._tag_list = [tl for j, tl in enumerate(self._tag_list) if j in self._indices]
            self._ne_annot_list = [
                nal for j, nal in enumerate(self._ne_annot_list) if j in self._indices
            ]
        return

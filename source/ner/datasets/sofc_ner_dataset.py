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
This module contains the SOFC NER dataset class.
"""

from typing import Tuple

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import BertTokenizer

from source.constants.mulms_constants import (
    sofc_bilou_id2ne_label,
    sofc_bilou_id2slot_label,
    sofc_bilou_ne_label2id,
    sofc_bilou_slot_label2id,
    sofc_ne_labels,
    sofc_slot_labels,
)
from source.relation_extraction.data_handling.sofc_relation_dataset import (
    SOFC_Relation_Dataset,
    load_sofc_relation_dataset,
)


class SOFC_NER_Dataset(TorchDataset):
    """
    This class provides an interface to the named entities of the SOFC corpus.
    """

    def __init__(self, split: str, bert_model_path: str, entity_type: str = "ENTITY") -> None:
        """


        Args:
            split (str): Desired split. Must be one of [train, validation, test].
            bert_model_path (str): Name or path of the BERT model used for tokenizing.
            entity_type (str, optional): Whether to use coarse-grained entities (ENTITY) or fine-grained entities (SLOT) that are provided by the SOFC corpus. Defaults to "ENTITY".
        """
        super().__init__()
        assert split in [
            "train",
            "validation",
            "test",
        ], "Invalid split provided. Split must be either train, validation or test."
        assert entity_type in [
            "ENTITY",
            "SLOT",
        ], "Type of named entities must be either ENTITY or SLOT."
        self._ent_type: str = "ne" if entity_type == "ENTITY" else "slot"
        self._split: str = split
        self.name: str = "SOFC_NER"
        self._MAX_LENGTH: int = 512
        self.ne_labels: list[str] = sofc_slot_labels
        self.bilou_ne_label2id: dict = sofc_bilou_slot_label2id
        self.bilou_id2ne_label: dict = sofc_bilou_id2slot_label
        # Override if using Entities instead of slots
        if self._ent_type == "ne":
            self.ne_labels = sofc_ne_labels
            self.bilou_ne_label2id = sofc_bilou_ne_label2id
            self.bilou_id2ne_label = sofc_bilou_id2ne_label
        self._ner_data: dict = {
            "id": [],
            "sentences": [],
            "tokens": [],
            "ne_labels": [],
            "ne_labels_bilou": [],
            "ne_labels_bilou_string": [],
            "slot_labels": [],
            "slot_labels_bilou": [],
            "slot_labels_bilou_string": [],
            "crf_mask": [],
            "tensor_encoded_input": None,
        }  # This dict contains the data unrolled s.t. it can be iterated over
        self._ne_annot_list: list[set[tuple]] = []
        self._slot_annot_list: list[set[tuple]] = []
        self._removable_labels: list[str] = ["interconnect_material"]
        self._load_sofc_dataset(bert_model_path)

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
            index (int): Index of the sample

        Returns:
            dict: Dict containing ID, test, BERT tensors, (sorted) CRF mask, label IDs and labels
        """
        return {
            "id": self._ner_data["id"][index],
            "text": self._ner_data["sentences"][index],
            "input_ids": self._ner_data["tensor_encoded_input"]["input_ids"][index],
            "attention_mask": self._ner_data["tensor_encoded_input"]["attention_mask"][index],
            "token_type_ids": self._ner_data["tensor_encoded_input"]["token_type_ids"][index],
            "crf_mask": self._ner_data["crf_mask"][index],
            "sorted_crf_mask": self._ner_data["sorted_crf_mask"][index],
            "label_ids": self._ner_data[f"{self._ent_type}_label_ids"][index],
            "label": str(
                self._ner_data[f"{self._ent_type}_labels"][index]
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
        if self._ent_type == "ne":
            return (
                self._ner_data["ne_labels_bilou"],
                self._ner_data["ne_labels_bilou_string"],
                self._ne_annot_list,
            )
        else:
            return (
                self._ner_data["slot_labels_bilou"],
                self._ner_data["slot_labels_bilou_string"],
                self._slot_annot_list,
            )

    def _load_sofc_dataset(self, bert_model_path: str) -> None:
        """
        Loads the SOFC corpus from the disk and prepares all tensors.

        Args:
            bert_model_path (str): BERT model used for the tokenizer.
        """
        # We can reuse that since it already contains all NER information
        sofc_relation_dataset: SOFC_Relation_Dataset = load_sofc_relation_dataset(self._split)

        data: dict = {}
        # Collect all relevant data
        for doc_id in range(len(sofc_relation_dataset._sentences)):
            data[doc_id] = {}
            for sent_id, sent in sofc_relation_dataset._sentences[doc_id].items():
                data[doc_id][sent_id] = {
                    "sentence": sent,
                    "named_entities": [],
                    "slots": [],
                    "ne_label_ids_bilou": [],
                    "slot_label_ids_bilou": [],
                    "ne_labels_bilou": [],
                    "slot_labels_bilou": [],
                }
                data[doc_id][sent_id]["tokens"] = sofc_relation_dataset._tokens[doc_id][sent_id]
                data[doc_id][sent_id]["ne_labels_bilou"] = ["O"] * len(
                    data[doc_id][sent_id]["tokens"]
                )
                data[doc_id][sent_id]["slot_labels_bilou"] = ["O"] * len(
                    data[doc_id][sent_id]["tokens"]
                )

            for entity in sofc_relation_dataset._named_entities[doc_id]:
                if entity.label not in self._removable_labels:
                    data[doc_id][entity.sent_id]["named_entities"].append(entity)
            for slot in sofc_relation_dataset._slots[doc_id]:
                if slot.label not in self._removable_labels:
                    data[doc_id][slot.sent_id]["slots"].append(slot)

        # Create BILOU labels
        B: str = "B-{0}"
        I: str = "I-{0}"
        L: str = "L-{0}"
        U: str = "U-{0}"
        for doc_id in range(len(data)):
            for sent_id in data[doc_id].keys():
                self._ne_annot_list.append(set())
                self._slot_annot_list.append(set())
                for ne in data[doc_id][sent_id]["named_entities"]:
                    label: str = ne.label
                    # The case that we need a U- Label
                    if (ne.end_idx - ne.begin_idx) == 0:
                        data[doc_id][sent_id]["ne_labels_bilou"][ne.begin_idx] = B.format(label)
                    else:
                        data[doc_id][sent_id]["ne_labels_bilou"][ne.begin_idx] = B.format(label)
                        j = ne.begin_idx + 1
                        while j < ne.end_idx:
                            data[doc_id][sent_id]["ne_labels_bilou"][j] = I.format(label)
                            j += 1
                        data[doc_id][sent_id]["ne_labels_bilou"][ne.end_idx] = I.format(label)

                    self._ne_annot_list[-1].add(tuple([ne.label, ne.begin_idx, ne.end_idx]))

                data[doc_id][sent_id]["ne_label_ids_bilou"] = [
                    sofc_bilou_ne_label2id[nl] for nl in data[doc_id][sent_id]["ne_labels_bilou"]
                ]

                for slot in data[doc_id][sent_id]["slots"]:
                    label: str = slot.label

                    if (slot.end_idx - slot.begin_idx) == 0:
                        data[doc_id][sent_id]["slot_labels_bilou"][slot.begin_idx] = U.format(
                            label
                        )
                    else:
                        data[doc_id][sent_id]["slot_labels_bilou"][slot.begin_idx] = B.format(
                            label
                        )
                        j = slot.begin_idx + 1
                        while j < slot.end_idx:
                            data[doc_id][sent_id]["slot_labels_bilou"][j] = I.format(label)
                            j += 1
                        data[doc_id][sent_id]["slot_labels_bilou"][slot.end_idx] = L.format(label)

                    self._slot_annot_list[-1].add(
                        tuple([slot.label, slot.begin_idx, slot.end_idx])
                    )

                data[doc_id][sent_id]["slot_label_ids_bilou"] = [
                    sofc_bilou_slot_label2id[sl]
                    for sl in data[doc_id][sent_id]["slot_labels_bilou"]
                ]

        # We can put everything into one large list now that we collected all named entities and slots across the data structures
        tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        id: int = 0
        crf_masks: list[list[int]] = []
        for doc_id in range(len(data)):
            for sent_id, entry in data[doc_id].items():
                if len(entry["named_entities"]) == 0:
                    continue
                self._ner_data["id"].append(id)
                self._ner_data["sentences"].append(entry["sentence"])
                self._ner_data["tokens"].append(entry["tokens"])
                self._ner_data["ne_labels"].append(entry["named_entities"])
                self._ner_data["ne_labels_bilou"].append(entry["ne_label_ids_bilou"])
                self._ner_data["ne_labels_bilou_string"].append(entry["ne_labels_bilou"])
                self._ner_data["slot_labels"].append(entry["slots"])
                self._ner_data["slot_labels_bilou"].append(entry["slot_label_ids_bilou"])
                self._ner_data["slot_labels_bilou_string"].append(entry["slot_labels_bilou"])

                crf_mask: list[int] = [0]  # [CLS]

                for t in entry["tokens"]:
                    ids: list[int] = tokenizer.encode(t)
                    if len(ids) >= 3:  # Skip if white space has been tokenized
                        crf_mask.extend(
                            [1] + [0] * (len(ids) - 3)
                        )  # WordPiece tokens - (actual token + [CLS] + [SEP])
                    else:
                        pass
                crf_mask.append(0)  # [SEP]

                crf_masks.append(crf_mask)
                id += 1

        self._ner_data["tensor_encoded_input"] = tokenizer.batch_encode_plus(
            self._ner_data["tokens"], is_split_into_words=True, return_tensors="pt", padding=True
        )
        pad_size: int = self._ner_data["tensor_encoded_input"]["input_ids"].shape[1]
        self._ner_data["crf_mask"] = torch.tensor(
            [c + [0] * (pad_size - len(c)) for c in crf_masks], dtype=torch.uint8
        )  # Padded CRF masks
        self._ner_data["sorted_crf_mask"] = torch.sort(
            self._ner_data["crf_mask"], dim=1, descending=True, stable=True
        )[
            0
        ]  # Used for cutting output of CRF layer
        self._ner_data["ne_label_ids"] = torch.tensor(
            [
                nl + [sofc_bilou_ne_label2id["x"]] * (pad_size - len(nl))
                for nl in self._ner_data["ne_labels_bilou"]
            ]
        )
        self._ner_data["slot_label_ids"] = torch.tensor(
            [
                sl + [sofc_bilou_slot_label2id["x"]] * (pad_size - len(sl))
                for sl in self._ner_data["slot_labels_bilou"]
            ]
        )

        # Removing empty entries
        self._ne_annot_list = [nal for nal in self._ne_annot_list if len(nal) > 0]
        self._slot_annot_list = [sal for sal in self._slot_annot_list if len(sal) > 0]
        return

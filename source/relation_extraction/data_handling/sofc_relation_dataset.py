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
We highly recommend loading this dataset from the pickle file since creation is really slow.
"""
import logging
import os
import pickle
import sys
from collections import namedtuple
from pathlib import Path

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from source.constants.mulms_constants import (
    SOFC_DATASET_READER_PATH,
    SOFC_PATH,
    sofc_ne_label2id,
    sofc_slot_label2id,
)
from source.relation_extraction.data_handling.mulms_relation_sentences import (
    MuLMSRelationSentence,
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

CURRENT_FILE_PATH: Path = Path(__file__).parent.absolute()

Entity: namedtuple = namedtuple(
    "Entity", ["doc_id", "sent_id", "begin_idx", "end_idx", "label", "label_id", "entity_id"]
)
Slot: namedtuple = namedtuple(
    "Slot", ["doc_id", "sent_id", "begin_idx", "end_idx", "label", "label_id", "slot_id"]
)
Experiment_Link: namedtuple = namedtuple(
    "Experiment_Link", ["doc_id", "exp_id", "gov_span_id", "dep_span_id", "label", "label_id"]
)
Span: namedtuple = namedtuple(
    "Span",
    [
        "doc_id",
        "span_id",
        "sent_id",
        "entity_label",
        "entity_label_id",
        "text",
        "begin_token",
        "end_token",
    ],
)


class SOFC_Relation_Dataset:
    def __init__(self, split: str) -> None:
        super().__init__()
        assert split in [
            "train",
            "validation",
            "test",
        ], "Invalid split provided. Split must be either train, validation or test."
        self._split: str = split
        self._sentences: dict = {}
        self._tokens: dict = {}
        self._measurement_labels: dict = {}
        # 7-tuples of the form (Doc_ID, Sent_ID, begin_idx, end_idx, label, label ID, entity ID)
        self._named_entities: dict = {}
        self._slots: dict = {}
        # These represent experiment frames of the form (Doc_ID, Exp_ID, Gov_Span_ID, Dep_Span_ID, Label, Label_ID)
        self._experiment_links: list[Experiment_Link] = []
        # This dict is used as lookup for span IDs per document with spans being of the form (Doc_ID, Span_ID, Sent_ID, Label, Label_ID, Text, Begin_Token, End_Token)
        self._spans: dict = {}
        self._load_sofc_dataset()

    def __len__(self) -> int:
        return len(self._experiment_links)

    def __getitem__(self, index) -> dict:
        return self._experiment_links[index]

    def _load_sofc_dataset(self) -> None:
        dataset: Dataset = load_dataset(
            SOFC_DATASET_READER_PATH.__str__(), data_dir=SOFC_PATH.__str__(), split=self._split
        )

        for doc_id in range(len(dataset)):
            self._sentences[doc_id] = {}
            self._tokens[doc_id] = {}
            self._measurement_labels[doc_id] = {}
            for sent_id in range(len(dataset["sentences"][doc_id])):
                # We stick to the counting start at 1 from the original dataset
                self._sentences[doc_id][sent_id + 1] = dataset["sentences"][doc_id][sent_id]
                self._tokens[doc_id][sent_id + 1] = dataset["tokens"][doc_id][sent_id]
                self._measurement_labels[doc_id][sent_id + 1] = dataset["sentence_labels"][doc_id][
                    sent_id
                ]

            for i, exp_id in enumerate(dataset["experiments"][doc_id]["experiment_id"]):
                gov_span_id: int = dataset["experiments"][doc_id]["span_id"][i]
                for j, label_id in enumerate(
                    dataset["experiments"][doc_id]["slots"][i]["frame_participant_label"]
                ):
                    dep_span_id: int = dataset["experiments"][doc_id]["slots"][i]["slot_id"][j]
                    label: str = (
                        dataset.features["experiments"]
                        .feature["slots"]
                        .feature["frame_participant_label"]
                        .names[label_id]
                    )
                    self._experiment_links.append(
                        Experiment_Link(
                            doc_id,
                            exp_id,
                            gov_span_id,
                            dep_span_id,
                            label,
                            sofc_slot_label2id[label],
                        )
                    )

            self._spans[doc_id] = {}
            for i, span_id in enumerate(dataset[doc_id]["spans"]["span_id"]):
                try:
                    label: str = (
                        dataset.features["spans"]
                        .feature["entity_label"]
                        .names[dataset[doc_id]["spans"]["entity_label"][i]]
                    )
                    label = "O" if label == "" else label
                    sent_id: int = dataset[doc_id]["spans"]["sentence_id"][i]
                    # We need to subtract 1 from the sentence ID since sentence counts starts at 1 which is not reflected in the HF dataset
                    begin_token_id: int = dataset["token_offsets"][doc_id]["offsets"][sent_id - 1][
                        "begin_char_offset"
                    ].index(dataset[doc_id]["spans"]["begin_char_offset"][i])
                    end_token_id: int = dataset["token_offsets"][doc_id]["offsets"][sent_id - 1][
                        "end_char_offset"
                    ].index(dataset[doc_id]["spans"]["end_char_offset"][i])
                    token_text: str = " ".join(
                        dataset["tokens"][doc_id][sent_id - 1][begin_token_id : end_token_id + 1]
                    )
                    self._spans[doc_id][span_id] = Span(
                        doc_id,
                        span_id,
                        sent_id,
                        label,
                        sofc_ne_label2id["EXPERIMENT" if label == "O" else label],
                        token_text,
                        begin_token_id,
                        end_token_id,
                    )
                except Exception:
                    # Remove all experiment links that depend on invalid spans
                    self._experiment_links: list = [
                        exp
                        for exp in self._experiment_links
                        if not (
                            (exp.gov_span_id == span_id or exp.dep_span_id == span_id)
                            and exp.doc_id == doc_id
                        )
                    ]

            named_entities: list[int] = dataset["entity_labels"][doc_id]
            slots: list[int] = dataset["slot_labels"][doc_id]

            bio_named_entities: list[str] = [
                [dataset.features["entity_labels"].feature.feature.names[ne] for ne in ne_list]
                for ne_list in named_entities
            ]
            bio_slots: list[str] = [
                [dataset.features["slot_labels"].feature.feature.names[slot] for slot in slot_list]
                for slot_list in slots
            ]
            self._named_entities[doc_id] = []
            self._slots[doc_id] = []
            meas_ids: list[int] = [
                k for (k, v) in self._measurement_labels[doc_id].items() if v == 1
            ]
            for i, mi in zip(range(len(bio_named_entities)), meas_ids):
                current_id: int = 1
                current_start_idx: int = -1
                current_end_idx: int = -1
                current_entity: str = None
                j: int = 0
                # Even though Entities and Slots span over the same tokens, we treat them separatly in two loops
                while j < len(bio_named_entities[i]):
                    if "B-" in bio_named_entities[i][j]:
                        current_start_idx = j
                        current_entity = bio_named_entities[i][j].split("-")[-1]
                        k = j + 1
                        while (
                            k < len(bio_named_entities[i])
                            and bio_named_entities[i][k] == f"I-{current_entity}"
                        ):
                            k += 1
                        current_end_idx = k - 1  # Correct for one step look-ahead
                        self._named_entities[doc_id].append(
                            Entity(
                                doc_id,
                                mi,
                                current_start_idx,
                                current_end_idx,
                                current_entity,
                                sofc_ne_label2id[current_entity],
                                current_id,
                            )
                        )
                        current_id += 1
                    j += 1

                j = 0
                current_id: int = 1
                while j < len(bio_slots[i]):
                    if "B-" in bio_slots[i][j]:
                        current_start_idx = j
                        current_entity = bio_slots[i][j].split("-")[-1]
                        k = j + 1
                        while k < len(bio_slots[i]) and bio_slots[i][k] == f"I-{current_entity}":
                            k += 1
                        current_end_idx = k - 1  # Correct for one step look-ahead
                        if current_entity in sofc_slot_label2id:
                            self._slots[doc_id].append(
                                Slot(
                                    doc_id,
                                    mi,
                                    current_start_idx,
                                    current_end_idx,
                                    current_entity,
                                    sofc_slot_label2id[current_entity],
                                    current_id,
                                )
                            )
                        current_id += 1
                    j += 1

        # Now remove all experiment links that span across multiple sentence since they can't be handled by the dependency parser
        removable_experiments: list[Experiment_Link] = []
        for exp in self._experiment_links:
            doc_id: int = exp.doc_id
            gov_span_id: int = exp.gov_span_id
            dep_span_id: int = exp.dep_span_id
            gov_span: Span = None
            dep_span: Span = None
            for i, span in self._spans[doc_id].items():
                if span.span_id == gov_span_id:
                    gov_span = span
                elif span.span_id == dep_span_id:
                    dep_span = span

                if gov_span is not None and dep_span is not None:
                    if gov_span.sent_id != dep_span.sent_id:
                        removable_experiments.append(exp)
                    break

        for exp in removable_experiments:
            logging.log(
                logging.WARNING,
                f"Removing experiment {exp} since it spans over multiple sentences.",
            )
            self._experiment_links.remove(exp)
        return


def load_sofc_relation_dataset(split: str) -> SOFC_Relation_Dataset:
    dataset: SOFC_Relation_Dataset = None
    if os.path.exists(
        f"{CURRENT_FILE_PATH.joinpath('./pickle_datasets').__str__()}/sofc_relation_dataset_{split}.pickle"
    ):
        with open(
            f"{CURRENT_FILE_PATH.joinpath('./pickle_datasets').__str__()}/sofc_relation_dataset_{split}.pickle",
            mode="rb",
        ) as f:
            dataset = pickle.load(f, fix_imports=True)
    else:
        dataset = SOFC_Relation_Dataset(split)
        with open(
            f"{CURRENT_FILE_PATH.joinpath('./pickle_datasets').__str__()}/sofc_relation_dataset_{split}.pickle",
            mode="wb",
        ) as f:
            pickle.dump(dataset, f, fix_imports=True)

    return dataset


def get_sofc_relsent_dataloader(split_name, model, batch_size, shuffle=True, num_workers=1):
    sofc_data = load_sofc_relation_dataset(split_name)

    sentence_data = dict()

    # Gather tokens
    for doc_id, doc_tokens in sofc_data._tokens.items():
        sentence_data[doc_id] = dict()
        for sent_id, tokens in doc_tokens.items():
            sentence_data[doc_id][sent_id] = dict()
            sentence_data[doc_id][sent_id]["tokens"] = tokens
            sentence_data[doc_id][sent_id]["spans"] = list()
            sentence_data[doc_id][sent_id]["relations"] = list()

    # Gather spans
    for doc_id, doc_spans in sofc_data._spans.items():
        for span_id, span in doc_spans.items():
            assert span_id == span.span_id
            assert span.doc_id == doc_id
            assert span.sent_id in sentence_data[span.doc_id]
            sentence_data[span.doc_id][span.sent_id]["spans"].append(span)

    # Gather links
    for rel in sofc_data._experiment_links:
        gov_span = sofc_data._spans[rel.doc_id][rel.gov_span_id]
        dep_span = sofc_data._spans[rel.doc_id][rel.dep_span_id]

        # assert gov_span.entity_label == "O"  # There are some exceptions to this, but only a handful
        assert gov_span.sent_id == dep_span.sent_id

        sentence_data[rel.doc_id][gov_span.sent_id]["relations"].append(rel)

    # Convert to Relsent format
    rel_sents = list()
    for doc_data in sentence_data.values():
        for sent_data in doc_data.values():
            rel_sents.append(sofc_to_relsent(sent_data))

    return DataLoader(
        rel_sents,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: MuLMSRelationSentence.batchify(x, model, factorized=model.factorized),
    )


def sofc_to_relsent(sentence_data):
    tokens = sentence_data["tokens"]
    named_entities = {"id": list(), "value": list(), "tokenIndices": list()}
    relations = {"ne_id_gov": list(), "ne_id_dep": list(), "label": list()}

    for span in sentence_data["spans"]:
        named_entities["id"].append(span.span_id)
        named_entities["value"].append(span.entity_label)
        named_entities["tokenIndices"].append((span.begin_token, span.end_token))

    for rel in sentence_data["relations"]:
        relations["ne_id_gov"].append(rel.gov_span_id)
        relations["ne_id_dep"].append(rel.dep_span_id)
        relations["label"].append(rel.label)

    return MuLMSRelationSentence(tokens, named_entities, relations)

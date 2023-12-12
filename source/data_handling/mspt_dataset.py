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
This module contains the dataset class for the MSPT dataset.
"""

import os
from collections import namedtuple

from puima.collection_utils import DocumentCollection

from source.constants.mulms_constants import (
    MSPT_PATH,
    mspt_ne_label2id,
    mspt_rel_label2id,
)

SENTENCE_TYPE: str = "webanno.custom.Sentence"
TOKEN_TYPE: str = "webanno.custom.Token"
RELATION_TYPE: str = "webanno.custom.Relation"
NE_TYPE: str = "webanno.custom.NamedEntity"

Sentence: namedtuple = namedtuple("Sentence", ["doc_id", "sent_id", "begin_offset", "end_offset"])
Token: namedtuple = namedtuple(
    "Token", ["doc_id", "sent_id", "token_id", "begin_idx", "end_idx", "text"]
)
Entity: namedtuple = namedtuple(
    "Entity",
    [
        "doc_id",
        "sent_id",
        "ent_id",
        "begin_idx",
        "end_idx",
        "begin_token",
        "end_token",
        "label",
        "label_id",
    ],
)
Relation: namedtuple = namedtuple(
    "Relation", ["doc_id", "sent_id", "rel_id", "gov_span_id", "dep_span_id", "label", "label_id"]
)


class MSPT_Dataset:
    """
    MSPT dataset class.
    """

    def __init__(self, split: str) -> None:
        """
        Initializes the MSPT dataset by reading it from the disk and preparing it for BERT-based training.

        Args:
            split (str): Desired split; must one of [ner-train, ner-dev, ner-test, sfex-train, sfex-dev, sfex-test]
        """
        assert split in [
            "ner-train",
            "ner-dev",
            "ner-test",
            "sfex-train",
            "sfex-dev",
            "sfex-test",
        ], "Invalid split provided. Split must be one of: ner-train, ner-dev, ner-test (NER), sfex-train, sfex-dev, sfex-test (Frames)"
        self._sentences: dict = {}  # Sentences sorted by Doc_ID
        self._sentences_as_string: dict = {}  # Sentences as string representation
        self._tokens: dict = {}  # Named Tuple Tokens sorted by Doc_ID and Sent_ID
        self._token_list: dict = {}  # Raw tokens sorted by Doc_ID and Sent_ID
        self._relations: dict = {}  # Named Tuple Relations sorted by Doc_ID and Sent_ID
        self._named_entities: dict = {}  # Named Tuple Entities sorted by Doc_ID and Sent_ID
        self._named_entities_by_id: dict = {}  # Named Tuple Entities sorted by their ID
        self._ner_data: dict = {
            "id": [],
            "sentences": [],
            "tokens": [],
            "ne_labels": [],
            "ne_labels_bilou": [],
            "slot_labels": [],
            "slot_labels_bilou": [],
            "crf_mask": [],
            "tensor_encoded_input": None,
        }  # This dict contains the data unrolled s.t. it can be iterated over
        self._split: str = split
        self._load_mspt_relation_dataset()

    def _load_mspt_relation_dataset(self) -> None:
        """
        Reads the dataset from the disk and creates BERT-based tensors.
        """

        doc_collection: DocumentCollection = DocumentCollection(
            xmi_dir=MSPT_PATH.__str__(), file_list=os.listdir(MSPT_PATH.__str__())
        )
        split_docs: list[str] = None
        with open(
            os.path.join(MSPT_PATH, f"../{self._split}-fnames.txt"), mode="r", encoding="utf-8"
        ) as f:
            split_docs = f.read().splitlines()

        for doc_id, doc in doc_collection.docs.items():
            if not doc_id.split(".")[0] in split_docs:
                continue
            sentences: list = list(doc.select_annotations(SENTENCE_TYPE))
            self._sentences[doc_id] = []
            self._sentences_as_string[doc_id] = {}
            self._tokens[doc_id] = {}
            self._token_list[doc_id] = {}
            self._relations[doc_id] = {}
            self._named_entities[doc_id] = {}

            for sent_id, sent in enumerate(sentences):
                self._sentences[doc_id].append(Sentence(doc_id, sent_id, sent.begin, sent.end))
                self._sentences_as_string[doc_id][sent_id] = doc.get_covered_text(sent)

                sent_tokens: list = list(doc.select_covered(TOKEN_TYPE, sent))
                sent_relations: list = list(doc.select_covered(RELATION_TYPE, sent))
                sent_entities: list = list(doc.select_covered(NE_TYPE, sent))

                self._tokens[doc_id][sent_id] = []
                self._token_list[doc_id][sent_id] = []
                self._relations[doc_id][sent_id] = []
                self._named_entities[doc_id][sent_id] = []

                for tok in sent_tokens:
                    self._tokens[doc_id][sent_id].append(
                        Token(
                            doc_id,
                            sent_id,
                            tok.id,
                            tok.begin,
                            tok.end,
                            tok.get_feature_value("value"),
                        )
                    )
                    self._token_list[doc_id][sent_id].append(tok.get_feature_value("value"))

                for ent in sent_entities:
                    covered_tokens: list = list(doc.select_covered(TOKEN_TYPE, ent))
                    covering_tokens: list = list(doc.select_covering(TOKEN_TYPE, ent))
                    min_begin_idx: int = -1
                    max_end_idx: int = -1
                    if len(covered_tokens) > 0:
                        min_begin_idx = min([ct.begin for ct in covered_tokens])
                        max_end_idx = max([ct.end for ct in covered_tokens])
                    # Fallback if NEs do not cover whole token
                    elif len(covering_tokens) > 0:
                        min_begin_idx = min([ct.begin for ct in covering_tokens])
                        max_end_idx = max([ct.end for ct in covering_tokens])

                    begin_token_idx: int = [
                        (i, t)
                        for i, t in enumerate(self._tokens[doc_id][sent_id])
                        if t.begin_idx == min_begin_idx
                    ][0][0]
                    end_token_idx: int = [
                        (i, t)
                        for i, t in enumerate(self._tokens[doc_id][sent_id])
                        if t.end_idx == max_end_idx
                    ][0][0]
                    label: str = ent.get_feature_value("value").replace(
                        "-", "_"
                    )  # Fix s.t. evaluation logic does not break
                    self._named_entities[doc_id][sent_id].append(
                        Entity(
                            doc_id,
                            sent_id,
                            ent.id,
                            ent.begin,
                            ent.end,
                            begin_token_idx,
                            end_token_idx,
                            label,
                            mspt_ne_label2id[label],
                        )
                    )
                    self._named_entities_by_id[ent.id] = Entity(
                        doc_id,
                        sent_id,
                        ent.id,
                        ent.begin,
                        ent.end,
                        begin_token_idx,
                        end_token_idx,
                        label,
                        mspt_ne_label2id[label],
                    )

                for rel in sent_relations:
                    # Get correct Token ID based on offsets
                    dep_tokens: list = list(doc.select_covered(NE_TYPE, rel))
                    dep_span = list(doc.select_covering(NE_TYPE, dep_tokens[0]))[0]
                    gov_span_id: int = int(rel.get_feature_value("governor"))
                    rel_label: str = rel.get_feature_value("relationType")

                    # Check if relation stays within same sentence

                    try:
                        dep_ent: Entity = self._named_entities_by_id[dep_span.id]
                        gov_ent: Entity = self._named_entities_by_id[gov_span_id]
                    except KeyError:
                        continue

                    if dep_ent.doc_id == gov_ent.doc_id and dep_ent.sent_id == gov_ent.sent_id:
                        self._relations[doc_id][sent_id].append(
                            Relation(
                                doc_id,
                                sent_id,
                                rel.id,
                                gov_span_id,
                                dep_span.id,
                                rel_label,
                                mspt_rel_label2id[rel_label],
                            )
                        )

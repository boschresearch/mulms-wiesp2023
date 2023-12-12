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
This module contains the relation dataset class for the MSPT corpus.
"""

from torch.utils.data import DataLoader

from source.data_handling.mspt_dataset import MSPT_Dataset
from source.relation_extraction.data_handling.mulms_relation_sentences import (
    MuLMSRelationSentence,
)


def get_mspt_relsent_dataloader(split_name, model, batch_size, shuffle=True, num_workers=1):
    mspt_data = MSPT_Dataset(split_name)

    sentence_data = dict()

    # Gather tokens
    for doc_id, doc_tokens in mspt_data._tokens.items():
        sentence_data[doc_id] = dict()
        for sent_id, tokens in doc_tokens.items():
            sentence_data[doc_id][sent_id] = dict()
            sentence_data[doc_id][sent_id]["tokens"] = tokens
            sentence_data[doc_id][sent_id]["spans"] = list()
            sentence_data[doc_id][sent_id]["relations"] = list()

    # Gather NEs
    for doc_id, doc_nes in mspt_data._named_entities.items():
        for sent_id, sent_nes in doc_nes.items():
            for ne in sent_nes:
                assert doc_id == ne.doc_id
                assert sent_id == ne.sent_id
                assert sent_id in sentence_data[doc_id]
                sentence_data[doc_id][sent_id]["spans"].append(ne)

    # Gather links
    for doc_id, doc_rels in mspt_data._relations.items():
        for sent_id, sent_rels in doc_rels.items():
            for rel in sent_rels:
                gov_span = mspt_data._named_entities_by_id[rel.gov_span_id]
                dep_span = mspt_data._named_entities_by_id[rel.dep_span_id]

                assert sent_id == rel.sent_id == gov_span.sent_id == dep_span.sent_id

                sentence_data[doc_id][rel.sent_id]["relations"].append(rel)

    # Convert to Relsent format
    rel_sents = list()
    for doc_data in sentence_data.values():
        for sent_data in doc_data.values():
            rel_sents.append(mspt_to_relsent(sent_data))

    return DataLoader(
        rel_sents,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: MuLMSRelationSentence.batchify(x, model, factorized=model.factorized),
    )


def mspt_to_relsent(sentence_data):
    tokens = [t.text for t in sentence_data["tokens"]]
    named_entities = {"id": list(), "value": list(), "tokenIndices": list()}
    relations = {"ne_id_gov": list(), "ne_id_dep": list(), "label": list()}

    for span in sentence_data["spans"]:
        named_entities["id"].append(span.ent_id)
        named_entities["value"].append(span.label)
        named_entities["tokenIndices"].append((span.begin_token, span.end_token))

    for rel in sentence_data["relations"]:
        relations["ne_id_gov"].append(rel.gov_span_id)
        relations["ne_id_dep"].append(rel.dep_span_id)
        relations["label"].append(rel.label)

    return MuLMSRelationSentence(tokens, named_entities, relations)

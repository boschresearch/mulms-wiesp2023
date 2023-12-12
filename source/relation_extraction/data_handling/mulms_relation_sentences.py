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

import torch


class MuLMSRelationSentence:
    def __init__(self, tokens, named_entities, relations, relation_matrix=None):
        self.tokens = tokens

        self.named_entities = list()
        for ner_id, label, (start_ix, end_ix) in zip(
            named_entities["id"], named_entities["value"], named_entities["tokenIndices"]
        ):
            self.named_entities.append(((start_ix, end_ix), label, ner_id))
        self.named_entities.sort(
            key=lambda x: x[0][0] * len(self.tokens) + x[0][1]
        )  # Sort by start index first, then end index

        ner_id_to_ix_map = {ner_id: ix for ix, (_, _, ner_id) in enumerate(self.named_entities)}

        # Relations are stored in an NxN matrix (where N = number of named entities).
        # For each pair of named entities, the matrix stores the relation label between them.
        if relation_matrix is not None:
            self.relation_matrix = relation_matrix
        else:
            assert relations is not None
            self.relation_matrix = [
                ["[null]" for j in range(len(self.named_entities))]
                for i in range(len(self.named_entities))
            ]
            for gov_id, dep_id, rel_label in zip(
                relations["ne_id_gov"], relations["ne_id_dep"], relations["label"]
            ):
                gov_ix = ner_id_to_ix_map[gov_id]
                dep_ix = ner_id_to_ix_map[dep_id]
                self.relation_matrix[gov_ix][dep_ix] = rel_label

    def __len__(self):
        return len(self.tokens)

    def get_tensorized_ner_pos(self, padded_ner_length):
        ne_pos = [0 for _ in range(padded_ner_length)]  # Pad with 0 for gather operation

        for ne_ix, ((start_ix, end_ix), _, _) in enumerate(self.named_entities):
            ne_pos[ne_ix] = start_ix * len(self.tokens) + end_ix

        return torch.tensor(ne_pos)

    def get_tensorized_ner_labels(self, ne_vocab, padded_ner_length):
        ne_lbls = [0 for _ in range(padded_ner_length)]  # Pad with 0 for gather operation

        for ne_ix, (_, lbl, _) in enumerate(self.named_entities):
            ne_lbls[ne_ix] = ne_vocab.token2ix(lbl)

        return torch.tensor(ne_lbls)

    def get_tensorized_num_ne(self):
        return torch.tensor(len(self.named_entities))

    def get_tensorized_relations(self, rel_vocab, padded_ner_length):
        rel_ix_matrix = [[-1 for j in range(padded_ner_length)] for i in range(padded_ner_length)]

        for i, row in enumerate(self.relation_matrix):
            for j, rel_lbl in enumerate(row):
                rel_ix_matrix[i][j] = rel_vocab.token2ix(rel_lbl)

        return torch.tensor(rel_ix_matrix)

    def get_tensorized_factorized_relations(self, rel_vocab, padded_ner_length):
        rel_existence_matrix = [
            [-1 for j in range(padded_ner_length)] for i in range(padded_ner_length)
        ]
        rel_lbl_ix_matrix = [
            [-1 for j in range(padded_ner_length)] for i in range(padded_ner_length)
        ]

        for i, row in enumerate(self.relation_matrix):
            for j, rel_lbl in enumerate(row):
                rel_existence_matrix[i][j] = 0 if rel_lbl == "[null]" else 1
                rel_lbl_ix_matrix[i][j] = (
                    -1 if rel_lbl == "[null]" else rel_vocab.token2ix(rel_lbl)
                )

        return torch.tensor(rel_existence_matrix), torch.tensor(rel_lbl_ix_matrix)

    @staticmethod
    def batchify(mulms_rel_sents, model, factorized=False):
        tokens_batch = [mmr_sent.tokens for mmr_sent in mulms_rel_sents]

        max_num_ner = max(len(mmr_sent.named_entities) for mmr_sent in mulms_rel_sents)

        num_ne_tensor = torch.stack(
            [mmr_sent.get_tensorized_num_ne() for mmr_sent in mulms_rel_sents]
        )
        ner_pos_tensor = torch.stack(
            [mmr_sent.get_tensorized_ner_pos(max_num_ner) for mmr_sent in mulms_rel_sents]
        )
        ner_labels_tensor = torch.stack(
            [
                mmr_sent.get_tensorized_ner_labels(model.ne_vocab, max_num_ner)
                for mmr_sent in mulms_rel_sents
            ]
        )

        if factorized:
            rel_existence_matrices, rel_lbl_ix_matrices = zip(
                *[
                    mmr_sent.get_tensorized_factorized_relations(model.rel_vocab, max_num_ner)
                    for mmr_sent in mulms_rel_sents
                ]
            )
            rel_existence_tensor = torch.stack(rel_existence_matrices)
            rel_lbl_ix_tensor = torch.stack(rel_lbl_ix_matrices)
            return (
                tokens_batch,
                mulms_rel_sents,
                {
                    "ner_pos": ner_pos_tensor,
                    "ner_labels": ner_labels_tensor,
                    "num_ne": num_ne_tensor,
                    "rel_existence": rel_existence_tensor,
                    "rel_lbls": rel_lbl_ix_tensor,
                },
            )
        else:
            rel_matrix_tensor = torch.stack(
                [
                    mmr_sent.get_tensorized_relations(model.rel_vocab, max_num_ner)
                    for mmr_sent in mulms_rel_sents
                ]
            )
            return (
                tokens_batch,
                mulms_rel_sents,
                {
                    "ner_pos": ner_pos_tensor,
                    "ner_labels": ner_labels_tensor,
                    "num_ne": num_ne_tensor,
                    "rel_matrix": rel_matrix_tensor,
                },
            )

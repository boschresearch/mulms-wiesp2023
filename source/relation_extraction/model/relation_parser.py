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

import json
from os.path import join
from pathlib import Path

import torch
from torch import nn

import source.relation_extraction.model.embeddings.transformer_wrappers as embeddings_module
from source.relation_extraction.data_handling.mulms_relation_sentences import (
    MuLMSRelationSentence,
)
from source.relation_extraction.model.bce_loss import BCEWithLogitsLossWithIgnore
from source.relation_extraction.vocab import BasicVocab

from .biaffine import DeepBiaffineScorer


class MuLMSRelationParser(nn.Module):
    def __init__(
        self,
        input_embeddings,
        ne_vocab,
        rel_vocab,
        ne_label_embedding_dim=64,
        span_repr_hidden_dim=1024,
        span_repr_out_dim=768,
        biaff_hidden_dim=768,
        arc_biaff_hidden_dim=768,
        lbl_biaff_hidden_dim=256,
        lbl_loss_scale=0.05,
        factorized=False,
    ):
        super().__init__()

        # Transformer embedding layer
        self.embed = input_embeddings

        # NE and relation vocabs
        self.ne_vocab = ne_vocab
        self.rel_vocab = rel_vocab

        # Named Entity label embedding layer
        self.ne_lbl_embed = nn.Embedding(len(self.ne_vocab), ne_label_embedding_dim)

        # Span representation MLP
        self.span_mlp = nn.Sequential(
            nn.Linear(
                2 * self.embed.embedding_dim + self.ne_lbl_embed.embedding_dim,
                span_repr_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(span_repr_hidden_dim, span_repr_out_dim),
            nn.ReLU(),
        )

        # Biaffine relation scorer(s)
        self.factorized = factorized
        if self.factorized:
            # Two classifiers: Edge existene and edge labels
            self.rel_arc_scorer = DeepBiaffineScorer(
                input1_size=span_repr_out_dim,
                input2_size=span_repr_out_dim,
                hidden_size=arc_biaff_hidden_dim,
                output_size=1,
            )
            self.rel_label_scorer = DeepBiaffineScorer(
                input1_size=span_repr_out_dim,
                input2_size=span_repr_out_dim,
                hidden_size=lbl_biaff_hidden_dim,
                output_size=len(self.rel_vocab),
            )
        else:
            # One classifier: Edge labels only (non-existence encoded as special label)
            self.edge_classifier = DeepBiaffineScorer(
                input1_size=span_repr_out_dim,
                input2_size=span_repr_out_dim,
                hidden_size=biaff_hidden_dim,
                output_size=len(self.rel_vocab),
            )

        # Cross-entropy loss
        if self.factorized:
            self.bce_loss = BCEWithLogitsLossWithIgnore()
            self.lbl_loss_scale = lbl_loss_scale
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    @classmethod
    def from_folder(cls, model_folder):
        model_folder = Path(model_folder)
        with open(model_folder / "config.json", "r") as config_json_file:
            model_config = json.load(config_json_file)["model"]

        # Transformer model
        transformer_model_class = getattr(embeddings_module, model_config["transformer_class"])
        transformer_model_path = model_config["transformer_model_path"]
        transformer_model_kwargs = model_config["transformer_model_kwargs"]
        transformer_model = transformer_model_class(
            model_path=transformer_model_path, **transformer_model_kwargs
        )

        # Vocabs
        ne_vocab = BasicVocab(vocab_filename=model_config["ne_vocab_path"])
        rel_vocab = BasicVocab(vocab_filename=model_config["rel_vocab_path"])

        # Create model
        model = cls(transformer_model, ne_vocab, rel_vocab, **model_config["model_kwargs"])

        # Load saved weights
        checkpoint = torch.load(model_folder / "model_best.pth", map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

        return model

    def save_config(self, save_dir, prefix=""):
        self.embed.save_config(save_dir, prefix=prefix + "/input_embeddings")
        self.ne_vocab.to_file(join(save_dir, prefix, "ne_vocab"))
        self.rel_vocab.to_file(join(save_dir, prefix, "rel_vocab"))

    def forward(
        self,
        input_sents,
        num_ne,
        ne_pos,
        ne_labels,
        mode="evaluation",
        targets=None,
        post_process=False,
    ):
        if mode == "training":
            assert self.training
            assert targets is not None
        elif mode == "validation":
            assert not self.training
            assert targets is not None
        else:
            assert not self.training

        # Get token embeddings for sentences
        embeddings_batch, true_seq_lengths = self.embed(input_sents)

        # Dimension checks, unpacking targets if necessary
        batch_size, padded_seq_len, embed_dim = embeddings_batch.shape
        assert batch_size == len(input_sents)
        assert embed_dim == self.embed.embedding_dim

        assert num_ne.shape[0] == ne_pos.shape[0] == ne_labels.shape[0] == batch_size
        assert ne_pos.shape[1] == ne_labels.shape[1]
        padded_num_ne = ne_pos.shape[1]

        # Stop early if there are no named entities in this batch: In this case, there is nothing to predict
        if padded_num_ne == 0:
            if self.factorized:
                raise
            else:
                mulms_rel_sents = self._assemble_rel_sents_unfact(
                    input_sents, num_ne, ne_pos, ne_labels, [None] * batch_size
                )
                if mode == "training" or mode == "validation":
                    dummy_loss = torch.tensor(0.00, requires_grad=True)
                    return mulms_rel_sents, dummy_loss
                else:
                    return mulms_rel_sents

        if targets is not None:
            if self.factorized:
                assert isinstance(targets, tuple)
                rel_existence_targets, rel_lbl_targets = targets
                assert rel_existence_targets.shape == rel_lbl_targets.shape
                assert rel_existence_targets.shape[0] == batch_size
                assert (
                    rel_existence_targets.shape[1]
                    == rel_existence_targets.shape[2]
                    == padded_num_ne
                )
            else:
                assert len(targets.shape) == 3
                assert targets.shape[0] == batch_size
                assert targets.shape[1] == targets.shape[2] == padded_num_ne

        # Get start and endpoint indices for Named Entities
        ne_startpoint_indices = torch.div(
            ne_pos, true_seq_lengths.unsqueeze(1), rounding_mode="floor"
        )
        ne_endpoint_indices = torch.remainder(ne_pos, true_seq_lengths.unsqueeze(1))

        # Get startpoint, endpoint, and label embeddings for Named Entities
        ne_startpoint_embeddings = torch.gather(
            embeddings_batch,
            1,
            ne_startpoint_indices.unsqueeze(-1).expand(-1, -1, embeddings_batch.shape[2]),
        )
        ne_endpoint_embeddings = torch.gather(
            embeddings_batch,
            1,
            ne_endpoint_indices.unsqueeze(-1).expand(-1, -1, embeddings_batch.shape[2]),
        )
        ne_label_embeddings = self.ne_lbl_embed(ne_labels)

        assert ne_startpoint_embeddings.shape == ne_endpoint_embeddings.shape
        assert (
            batch_size
            == ne_startpoint_embeddings.shape[0]
            == ne_endpoint_embeddings.shape[0]
            == ne_label_embeddings.shape[0]
        )
        assert (
            padded_num_ne
            == ne_startpoint_embeddings.shape[1]
            == ne_endpoint_embeddings.shape[1]
            == ne_label_embeddings.shape[1]
        )

        # Concatenate and run through MLP to get Named Entity embeddings
        ne_concatenated = torch.cat(
            [ne_startpoint_embeddings, ne_endpoint_embeddings, ne_label_embeddings], dim=-1
        )
        ne_embeddings = self.span_mlp(ne_concatenated)

        # Run biaffine edge scorer(s) on Named Entity embeddings
        if self.factorized:
            rel_arc_scores = self.rel_arc_scorer(ne_embeddings, ne_embeddings).squeeze(dim=-1)
            rel_lbl_scores = self.rel_label_scorer(ne_embeddings, ne_embeddings)

            if (
                rel_arc_scores.numel() == 0
            ):  # No named entities in batch and therefore no edge predictions: Create dummy tensor
                rel_arc_scores = torch.zeros((batch_size, 1, 1))
            if rel_lbl_scores.numel() == 0:
                rel_lbl_scores = torch.zeros((batch_size, 1, 1, len(self.rel_vocab)))

            if mode == "training" or mode == "validation":
                assert rel_arc_scores.shape == rel_existence_targets.shape
                assert rel_lbl_scores.shape[0:3] == rel_lbl_targets.shape

                arc_loss = self.bce_loss(rel_arc_scores, rel_existence_targets)
                lbl_loss = self.ce_loss(
                    rel_lbl_scores.view(batch_size * padded_num_ne * padded_num_ne, -1),
                    rel_lbl_targets.view(batch_size * padded_num_ne * padded_num_ne),
                )
                loss = arc_loss + self.lbl_loss_scale * lbl_loss

                mulms_rel_sents = self._assemble_rel_sents_fact(
                    input_sents, num_ne, ne_pos, ne_labels, rel_arc_scores, rel_lbl_scores
                )
                return mulms_rel_sents, loss

            else:  # Evaluation: No loss computation
                mulms_rel_sents = self._assemble_rel_sents_fact(
                    input_sents, num_ne, ne_pos, ne_labels, rel_arc_scores, rel_lbl_scores
                )
                return mulms_rel_sents

        else:  # Unfactorized model
            edge_scores = self.edge_classifier(ne_embeddings, ne_embeddings)

            if (
                edge_scores.numel() == 0
            ):  # No named entities in batch and therefore no edge predictions: Create dummy tensor
                edge_scores = torch.zeros((batch_size, 1, 1, len(self.rel_vocab)))

            if mode == "training" or mode == "validation":
                assert edge_scores.shape[0:3] == targets.shape
                loss = self.ce_loss(
                    edge_scores.view(batch_size * padded_num_ne * padded_num_ne, -1),
                    targets.view(batch_size * padded_num_ne * padded_num_ne),
                )
                mulms_rel_sents = self._assemble_rel_sents_unfact(
                    input_sents, num_ne, ne_pos, ne_labels, edge_scores
                )
                return mulms_rel_sents, loss
            else:  # Evaluation: No loss computation
                mulms_rel_sents = self._assemble_rel_sents_unfact(
                    input_sents, num_ne, ne_pos, ne_labels, edge_scores
                )
                return mulms_rel_sents

    def _assemble_rel_sents_fact(
        self,
        input_sents,
        num_ne_batch,
        ne_pos_batch,
        ne_labels_batch,
        rel_arc_logits_batch,
        rel_lbl_logits_batch,
    ):
        assert (
            len(input_sents)
            == len(num_ne_batch)
            == len(ne_pos_batch)
            == len(ne_labels_batch)
            == len(rel_arc_logits_batch)
            == len(rel_lbl_logits_batch)
        )

        rel_sents = list()
        for tokens, num_ne, ne_pos, ne_labels, rel_arc_logits, rel_lbl_logits in zip(
            input_sents,
            num_ne_batch,
            ne_pos_batch,
            ne_labels_batch,
            rel_arc_logits_batch,
            rel_lbl_logits_batch,
        ):
            sent_length = len(tokens)

            ne_pos = ne_pos[:num_ne]
            ne_startpoints = ne_pos // sent_length
            ne_endpoints = ne_pos % sent_length
            ne_labels = ne_labels[:num_ne]

            named_entities = dict()
            named_entities["id"] = list(range(len(ne_labels)))
            named_entities["value"] = [
                self.ne_vocab.ix2token(int(ne_lbl_ix)) for ne_lbl_ix in ne_labels
            ]
            named_entities["tokenIndices"] = list(
                zip(ne_startpoints.tolist(), ne_endpoints.tolist())
            )

            rel_arc_logits_no_padding = rel_arc_logits[:num_ne, :num_ne]
            rel_lbl_logits_no_padding = rel_lbl_logits[:num_ne, :num_ne, :]
            rel_lbl_ixs = torch.argmax(rel_lbl_logits_no_padding, dim=-1)

            relation_matrix = [["[null]" for j in range(num_ne)] for i in range(num_ne)]
            for i in range(num_ne):
                for j in range(num_ne):
                    edge_exists = rel_arc_logits_no_padding[i, j] > 0
                    if edge_exists:
                        relation_matrix[i][j] = self.rel_vocab.ix2token(int(rel_lbl_ixs[i, j]))

            mulms_rel_sent = MuLMSRelationSentence(
                tokens, named_entities, None, relation_matrix=relation_matrix
            )

            rel_sents.append(mulms_rel_sent)

        return rel_sents

    def _assemble_rel_sents_unfact(
        self, input_sents, num_ne_batch, ne_pos_batch, ne_labels_batch, relation_logits_batch
    ):
        assert (
            len(input_sents)
            == len(num_ne_batch)
            == len(ne_pos_batch)
            == len(ne_labels_batch)
            == len(relation_logits_batch)
        )

        rel_sents = list()
        for tokens, num_ne, ne_pos, ne_labels, relation_logits in zip(
            input_sents, num_ne_batch, ne_pos_batch, ne_labels_batch, relation_logits_batch
        ):
            sent_length = len(tokens)

            ne_pos = ne_pos[:num_ne]
            ne_startpoints = ne_pos // sent_length
            ne_endpoints = ne_pos % sent_length
            ne_labels = ne_labels[:num_ne]

            named_entities = dict()
            named_entities["id"] = list(range(len(ne_labels)))
            named_entities["value"] = [
                self.ne_vocab.ix2token(int(ne_lbl_ix)) for ne_lbl_ix in ne_labels
            ]
            named_entities["tokenIndices"] = list(
                zip(ne_startpoints.tolist(), ne_endpoints.tolist())
            )

            if num_ne == 0:
                relation_matrix = []
            else:
                relation_logits_no_padding = relation_logits[:num_ne, :num_ne, :]
                relation_lbl_ixs = torch.argmax(relation_logits_no_padding, dim=-1)
                relation_matrix = [
                    [self.rel_vocab.ix2token(int(elem)) for elem in row]
                    for row in relation_lbl_ixs
                ]

            mulms_rel_sent = MuLMSRelationSentence(
                tokens, named_entities, None, relation_matrix=relation_matrix
            )

            rel_sents.append(mulms_rel_sent)

        return rel_sents

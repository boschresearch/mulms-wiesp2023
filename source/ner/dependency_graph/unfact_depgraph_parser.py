#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

"""
This module contains the parser for the unfactorized dependency graphs.
"""

import torch
from torch import nn

import source.ner.modules as modules_module
from source.ner.dependency_graph.unfact_depgraph import UnfactorizedDependencyGraph


class UnfactorizedDependencyGraphParser(nn.Module):
    def __init__(self, input_embeddings, label_scorer, do_lexicalization=True):
        super(UnfactorizedDependencyGraphParser, self).__init__()
        self.embed = input_embeddings
        self.label_scorer = label_scorer
        self.do_lexicalization = do_lexicalization

        self.root_embedding = nn.Parameter(
            torch.randn(input_embeddings.embedding_dim), requires_grad=True
        )

        self.labels_vocab = self.label_scorer.vocab

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    @classmethod
    def from_args_dict(cls, args_dict, model_dir=None):
        assert "input_embeddings" in args_dict.keys() and "label_scorer" in args_dict.keys()

        if "do_lexicalization" not in args_dict.keys():
            args_dict["do_lexicalization"] = True

        input_embeddings_dir = model_dir / "input_embeddings" if model_dir else None
        input_embeddings = getattr(
            modules_module, args_dict["input_embeddings"]["type"]
        ).from_args_dict(args_dict["input_embeddings"]["args"], model_dir=input_embeddings_dir)

        args_dict["label_scorer"]["args"]["input_dim"] = input_embeddings.embedding_dim

        label_scorer_dir = model_dir / "label_scorer" if model_dir else None
        label_scorer = getattr(modules_module, args_dict["label_scorer"]["type"]).from_args_dict(
            args_dict["label_scorer"]["args"], model_dir=label_scorer_dir
        )

        return cls(
            input_embeddings, label_scorer, do_lexicalization=args_dict["do_lexicalization"]
        )

    def save_config(self, save_dir, prefix=""):
        self.embed.save_config(save_dir, prefix=prefix + "/input_embeddings")
        self.label_scorer.save_config(save_dir, prefix=prefix + "/label_scorer")

    def forward(
        self,
        input_sents,
        mode="evaluation",
        targets=None,
        ix_to_id=None,
        multiword_tokens=None,
        return_logits=False,
        post_process=True,
    ):
        if mode == "training":
            assert self.training
            assert targets is not None
        elif mode == "validation":
            assert not self.training
            assert targets is not None
        else:
            assert not self.training

        # Dimension check: Batch size
        batch_size = len(input_sents)
        if targets is not None:
            assert batch_size == targets.shape[0]

        embeddings_batch, true_seq_lengths = self.embed(input_sents)

        padded_seq_len = embeddings_batch.shape[1]
        if targets is not None:
            assert padded_seq_len**2 == targets.shape[1]

        # Compute dependency labels (run label scorer)
        labels_logits = self.label_scorer(embeddings_batch)

        if (
            mode == "training" or mode == "validation"
        ):  # When training or validating: Compute losses
            labels_logits_flat = labels_logits.view(batch_size * padded_seq_len**2, -1)
            gold_labels_flat = targets.view(batch_size * padded_seq_len**2)
            labels_loss = self.ce_loss(labels_logits_flat, gold_labels_flat)

            # Do not perform greedy heuristic algorithm during training or validation
            predicted_depgraphs = self._assemble_depgraphs(
                input_sents, labels_logits, post_process=post_process
            )

            return predicted_depgraphs, labels_loss
        else:  # When evaluating: Only compute graphs, not loss
            predicted_depgraphs = self._assemble_depgraphs(
                input_sents,
                labels_logits,
                post_process=post_process,
                ix_to_id=ix_to_id,
                multiword_tokens=multiword_tokens,
            )

            if return_logits:
                return predicted_depgraphs, labels_logits
            else:
                return predicted_depgraphs

    def _assemble_depgraphs(self, input_sents, labels_logits_batch, post_process=True):
        assert len(input_sents) == len(labels_logits_batch)

        label_ix_batch = torch.argmax(labels_logits_batch, dim=-1)

        graphs = list()
        for tokens, dep_label_ixs in zip(input_sents, label_ix_batch):
            assert len(dep_label_ixs.shape) == 2
            sent_length = len(tokens)

            dep_labels_no_padding = dep_label_ixs[:sent_length, :sent_length]  # Cut off padding
            dependencies = [
                [self.labels_vocab.ix2token(int(elem)) for elem in row]
                for row in dep_labels_no_padding
            ]

            graphs.append(UnfactorizedDependencyGraph(tokens, dependencies))

        return graphs

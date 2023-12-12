#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

"""
This module contains the unfactorized dependency graph data structure.
"""

import itertools

import numpy as np
import torch

from source.constants.mulms_constants import mulms_ne_dependency_labels


class UnfactorizedDependencyGraph:
    """
    The unfactorized dependency graph class used for dependency parsing. For more information, please refer to https://github.com/boschresearch/steps-parser.
    """

    def __init__(
        self, tokens, dependencies, gold_labels=None, ix_to_id=None, multiword_tokens=None
    ):
        """
        Initializes the unfactorized dependency graph.
        """
        self.tokens = tokens

        if not multiword_tokens:
            self.multiword_tokens = dict()
        else:
            self.multiword_tokens = multiword_tokens

        if not ix_to_id:
            self.ix_to_id = {i: str(i) for i in range(len(tokens))}
        else:
            self.ix_to_id = ix_to_id

        self.dependencies = dependencies
        self._gold_labels: set[tuple] = gold_labels

        assert len(self.dependencies) == len(self.tokens)

    def __len__(self):
        """Return the number of tokens underlying this dependency graph. Note that this does not include the
        artificial [root] token that is added to every graph."""
        return len(self.tokens)

    @classmethod
    def load_from_dataset(cls, data: list[np.ndarray]):
        """
        Creates an Dependency Graph given an array which contains IDs, list of tokens and NE labels.

        Args:
            data (list[np.ndarray]): List of numpy arrays (IDs, Tokens, NE Labels)

        Returns:
            UnfactorizedDependencyGraph: Dependency Graph Instance
        """
        tokens: list[str] = data[1].tolist()
        dependencies: list[list[str]] = [
            ["O" for _ in range(len(tokens))] for _ in range(len(tokens))
        ]
        gold_labels: set(tuple) = set()

        for id, label in zip(data[0], data[2]):

            if label != "O":
                curr_incoming_edge_annots: list = label.split("|")

                for edge in curr_incoming_edge_annots:
                    gov, annot = tuple(edge.split(":"))
                    gold_labels.add((annot, int(gov) - 1, id - 1))
                    if dependencies[int(gov) - 1][id - 1] != "O":
                        if annot == dependencies[int(gov) - 1][id - 1]:
                            continue
                        if (
                            f"{annot}+{dependencies[int(gov)-1][id -1]}"
                            in mulms_ne_dependency_labels
                        ):
                            dependencies[int(gov) - 1][
                                id - 1
                            ] = f"{annot}+{dependencies[int(gov)-1][id -1]}"
                        elif (
                            f"{dependencies[int(gov)-1][id -1]}+{annot}"
                            in mulms_ne_dependency_labels
                        ):
                            dependencies[int(gov) - 1][
                                id - 1
                            ] = f"{dependencies[int(gov)-1][id -1]}+{annot}"
                        else:
                            continue
                    else:
                        dependencies[int(gov) - 1][id - 1] = annot

        return cls(tokens, dependencies, gold_labels)

    def pretty_print(self):
        """Display this dependency matrix as a nicely formatted table.

        Args:
            tokens: The tokens of the sentence.
        """
        tokens = self.tokens

        # Determine required column width for printing
        col_width = 0
        for token in tokens:
            col_width = max(col_width, len(token))
        for i in range(len(self)):
            for j in range(len(self.dependencies[i])):
                if len(self.dependencies[i][j]) > col_width:
                    col_width = max(col_width, len(self.dependencies[i][j]))
        col_width += 3

        # Print dependency matrix
        print()
        print("".join(token.rjust(col_width) for token in [""] + tokens))
        print()
        for head_ix in range(len(tokens)):
            print(tokens[head_ix].rjust(col_width), end="")
            for dependent_ix in range(len(tokens)):
                print(self.dependencies[head_ix][dependent_ix].rjust(col_width), end="")
            print()
            print()

    def get_tensorized_dependency_matrix(self, vocab, padded_length):
        """
        Returns the dependency matrix as flattened tensor.
        """
        assert padded_length >= len(self.dependencies)

        label_ixs: list[list[int]] = [
            [-1 for j in range(padded_length)] for i in range(padded_length)
        ]

        for head_ix in range(len(self.dependencies)):
            for dep_ix in range(len(self.dependencies)):
                label_ixs[head_ix][dep_ix] = vocab.token2ix(self.dependencies[head_ix][dep_ix])

        label_ixs_flat = list(itertools.chain(*label_ixs))

        return torch.tensor(label_ixs_flat)

    @staticmethod
    def batchify(depgraphs, model):
        """Tensorize and batchify a list of UnfactorizedDependencyGraph objects for use with the given model.

        Args:
            depgraphs: A list of UnfactorizedDependencyGraph objects.
            model: The model instance to create tensors for.

        Returns:
            A 3-tuple consisting of the list of underlying sentences, a list of the dependency trees themselves, and the
            gold annotations in batched tensor form.
        """
        sentences = [graph.tokens for graph in depgraphs]
        max_sent_length = max(len(sent) for sent in sentences)

        deps_tensor = torch.stack(
            [
                graph.get_tensorized_dependency_matrix(model.labels_vocab, max_sent_length)
                for graph in depgraphs
            ]
        )

        return sentences, depgraphs, deps_tensor

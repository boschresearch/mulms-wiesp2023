#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from os import makedirs
from os.path import join

import torch.nn.functional as F
from torch import nn

import source.relation_extraction.model.biaffine as scorer_module
import source.relation_extraction.vocab as vocab_module


class DependencyClassifier(nn.Module):
    """Module for classifying (syntactic/semantic) dependencies between pairs of tokens. Every token pair is mapped
    onto a logits vector with the dimensionality of the specified output label vocabulary. Label indices are then
    extracted from these logits.
    """

    def __init__(self, input_dim, vocab, scorer_class, hidden_size, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of input vectors.
            vocab: Vocabulary of output labels.
            scorer_class: Which class to use for scoring arcs, e.g. DeepBiaffineScorer.
            hidden_size: Hidden size of the scorer module.
            dropout: Dropout ratio of the scorer module. Default: 0.
        """
        super(DependencyClassifier, self).__init__()
        self.vocab = vocab
        self.scorer = getattr(scorer_module, scorer_class)(
            input1_size=input_dim,
            input2_size=input_dim,
            hidden_size=hidden_size,
            output_size=len(self.vocab),
            hidden_func=F.relu,
            dropout=dropout,
        )

    @classmethod
    def from_args_dict(cls, args_dict, model_dir=None):
        if "args" not in args_dict["vocab"]:
            args_dict["vocab"]["args"] = dict()

        if args_dict["vocab"]["type"] == "BasicVocab" and model_dir is not None:
            args_dict["vocab"]["args"]["vocab_filename"] = model_dir / "vocab"

        args_dict["vocab"] = getattr(vocab_module, args_dict["vocab"]["type"])(
            **args_dict["vocab"]["args"]
        )

        return cls(**args_dict)

    def save_config(self, save_dir, prefix=""):
        if isinstance(self.vocab, vocab_module.BasicVocab):
            makedirs(join(save_dir, prefix), exist_ok=True)
            self.vocab.to_file(join(save_dir, prefix, "vocab"))

    def forward(self, embeddings_batch, embeddings2_batch=None):
        """Take a batch of embedding sequences and feed them to the classifier to obtain logits for dependency labels
        for each pair of tokens.

        Args:
            embeddings_batch: Tensor (shape: batch_size * max_seq_len * embeddings_dim) containing input embeddings.
            embeddings2_batch: Tensor (shape: batch_size * max_seq_len_2 * embeddings_dim_2) containing input
              embeddings. If None, embeddings_batch will be used twice.

        Returns: A tuple consisting of (a) a tensor containing the output logits of the dependency classifier
          ("flattened"; shape: `batch_size * (max_seq_len**2) * vocab_size`); (b) a tensor containing the actual
          predictions, in the form of label indices ("flattened"; shape: `batch_size * (max_seq_len**2)`).
        """
        batch_size = embeddings_batch.shape[0]  # noqa: F841
        seq_len = embeddings_batch.shape[1]  # noqa: F841

        # Run the scorer on all the token pairs
        if embeddings2_batch is None:
            logits = self.scorer(embeddings_batch, embeddings_batch)
        else:
            logits = self.scorer(embeddings_batch, embeddings2_batch)

        # Return the logits
        return logits

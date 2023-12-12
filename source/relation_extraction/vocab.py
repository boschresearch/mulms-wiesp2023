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


class BasicVocab:
    """Class for mapping labels/tokens to indices and vice versa."""

    def __init__(
        self,
        vocab_filename=None,
        ignore_label="__IGNORE__",
        ignore_index=-1,
        load_from_disk: bool = True,
    ):
        """A vocabulary is read from a file in which each label constitutes one line. The index associated with each
        label is the index of the line that label occurred in (counting from 0).

        In addition, a special label and index (`ignore_label` and `ignore_index`) are added to signify content which
        should be ignored in parsing/tagging tasks.

        Args:
            vocab_filename: Name of the file to read the vocabulary from.
            ignore_label: Special label signifying ignored content. Default: `__IGNORE__`.
            ignore_index: Special index signifying ignored content. Should be negative to avoid collisions with "true"
              indices. Default: `-1`.
        """
        self.ix2token_data = dict()
        self.token2ix_data = dict()

        self.vocab_filename = vocab_filename

        if self.vocab_filename is not None:
            if load_from_disk:
                with open(str(vocab_filename)) as vocab_file:
                    vocab = vocab_file.read().split("\n")
            else:
                vocab = self.vocab_filename
            for ix, line in enumerate(vocab):
                token = line.strip()

                self.ix2token_data[ix] = token
                self.token2ix_data[token] = ix

        self.ignore_label = ignore_label
        self.ignore_index = ignore_index

        self.ix2token_data[ignore_index] = ignore_label
        self.token2ix_data[ignore_label] = ignore_index

        assert self.is_consistent()

    def __len__(self) -> int:
        """
        Returns the size of the vocab.

        Returns:
            int: Length of the vocab
        """
        return len(self.ix2token_data) - 1  # Do not count built-in "ignore" label

    def __str__(self) -> str:
        """
        Returns the list of tokens in the vocab as string.

        Returns:
            str: List of tokens in the vocab
        """
        # Do not consider built-in "ignore" label
        return "\n".join(
            self.ix2token_data[ix] for ix in sorted(self.ix2token_data.keys()) if ix >= 0
        )

    def __contains__(self, key: str) -> bool:
        """
        Checks if key is contained in vocab.

        Args:
            key (str): Lookup key

        Returns:
            bool: Key is contained or not.
        """
        return key in self.token2ix_data

    def ix2token(self, ix) -> str:
        """Get the token associated with index `ix`."""
        return self.ix2token_data[ix]

    def token2ix(self, token) -> int:
        """Get the index associated with token `token`."""
        return self.token2ix_data[token]

    def add(self, token) -> None:
        """Adds a token to the vocabulary if it does not already exist."""
        if token not in self.token2ix_data:
            new_ix = len(self)

            self.token2ix_data[token] = new_ix
            self.ix2token_data[new_ix] = token

    def to_file(self, vocab_filename: str) -> None:
        """Write vocabulary to a file."""
        with open(vocab_filename, "w") as vocab_file:
            vocab_file.write(str(self))

    def is_consistent(self) -> bool:
        """Checks if all index mappings match up. Used for debugging."""
        if len(self.ix2token_data) != len(self.token2ix_data):
            return False

        try:
            for token, ix in self.token2ix_data.items():
                if self.ix2token_data[ix] != token:
                    return False
        except IndexError:
            return False

        if "[null]" in self.token2ix_data:
            assert self.token2ix_data["[null]"] == 0

        return True

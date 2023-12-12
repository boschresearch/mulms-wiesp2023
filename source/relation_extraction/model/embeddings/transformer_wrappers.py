#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import random
from os import makedirs
from os.path import join

import torch
from torch import nn
from torch.nn import Dropout
from transformers import BertConfig, BertModel, BertTokenizer


class ScalarMixWithDropout(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of the dimensions of a tensor, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.
    If ``dropout > 0``, then for each scalar weight, adjust its softmax weight mass to 0 with
    the dropout probability (i.e., setting the unnormalized weight to -inf). This effectively
    should redistribute dropped probability mass to all other weights.
    """

    def __init__(
        self,
        mixture_size,
        trainable=True,
        initial_scalar_parameters=None,
        layer_dropout=None,
        layer_dropout_value=-1e20,
    ):
        """
        Args:
            mixture_size: Number of layers to mix.
            trainable: Whether to train the weights of the scalar mixture. Default: True.
            initial_scalar_parameters: Initial parameters (un-normalized weights) of the scalar mixture. If not
              provided, all initial weights are set to 0. Default: None.
            layer_dropout: Dropout ratio for entire layers of scalar mixture. Default: None.
            layer_dropout_value: Value to replace the unnormalized weight of dropped layers with. Should be "close" to
              negative infinity. Default: -1e20.
        """
        super(ScalarMixWithDropout, self).__init__()
        self.mixture_size = mixture_size
        self.layer_dropout = layer_dropout

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size

        assert len(initial_scalar_parameters) == mixture_size

        self.scalar_parameters = nn.Parameter(
            torch.FloatTensor(initial_scalar_parameters), requires_grad=trainable
        )
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

        if self.layer_dropout:
            layer_dropout_mask = torch.zeros(len(self.scalar_parameters))
            layer_dropout_fill = torch.empty(len(self.scalar_parameters)).fill_(
                layer_dropout_value
            )
            self.register_buffer("layer_dropout_mask", layer_dropout_mask)
            self.register_buffer("layer_dropout_fill", layer_dropout_fill)

    def forward(self, input_tensor):
        """Compute a weighted sum of the dimensions of ``input_tensor`` using the coefficients stored within the
        module.
        """
        assert input_tensor.shape[0] == self.mixture_size
        num_dim = len(input_tensor.shape)

        if self.layer_dropout and self.training:
            weights = torch.where(
                self.layer_dropout_mask.uniform_() > self.layer_dropout,
                self.scalar_parameters,
                self.layer_dropout_fill,
            )
        else:
            weights = self.scalar_parameters

        normed_weights = torch.nn.functional.softmax(weights, dim=0)
        normed_weights = normed_weights[
            (...,) + (None,) * (num_dim - 1)
        ]  # Unsqueeze weight tensor for proper broadcasting

        return self.gamma * torch.sum(input_tensor * normed_weights, dim=0)


class TransformerWrapper(nn.Module):
    """Base class for turning batches of sentences into tensors of (BERT/RoBERTa/...) embeddings.

    An object of this class takes as input a bunches of sentences (represented as a lists of lists of tokens) and
    returns, for each specified output ID, tensors (shape: batch_size * max_sent_len * embedding_dim) of token
    embeddings. The embeddings for the different outputs are generated using the same underlying transformer model, but
    by default use different scalar mixtures of the internal transformer layers to generate final embeddings.
    """

    def __init__(
        self,
        model_class,
        tokenizer_class,
        config_class,
        model_path,
        output_ids=None,
        tokenizer_path=None,
        config_only=False,
        fine_tune=True,
        shared_embeddings=None,
        hidden_dropout=0.2,
        attn_dropout=0.2,
        output_dropout=0.5,
        scalar_mix_layer_dropout=0.1,
        token_mask_prob=0.2,
        word_piece_pooling="first",
    ):
        """
        Args:
            model_class: Class of transformer model to use for token embeddings.
            tokenizer_class: Class of tokenizer to use for tokenization.
            config_class: Class of transformer config.
            model_path: Path to load transformer model from.
            output_ids: List of output IDs to generate embeddings for. These outputs will get separately trained
              scalar mixtures. If none provided, there will be only one scalar mix and one output.
            tokenizer_path: Path to load tokenizer from (default: None; specify when using config_only option).
            config_only: If True, only load model config, not weights (default: False).
            fine_tune: Whether to fine-tune the transformer language model. If False, weights of the transformer model
              will not be trained. Default: True.
            shared_embeddings: If specified (as list of lists of output IDs), the specified groups of outputs will
              share the same scalar mixture (and thus embeddings). Default: None.
            hidden_dropout: Dropout ratio for hidden layers of the transformer model.
            attn_dropout: Dropout ratio for the attention probabilities.
            output_dropout: Dropout ratio for embeddings output.
            scalar_mix_layer_dropout: Dropout ratio for transformer layers.
            token_mask_prob: Probability of replacing input tokens with mask token.
            word_piece_pooling: How to combine multiple word piece embeddings into one token embedding. Default: "first".
        """
        super(TransformerWrapper, self).__init__()

        if not output_ids:
            self.output_ids = ["__dummy_output__"]
        else:
            self.output_ids = output_ids

        self.model, self.tokenizer = self._init_model(
            model_class,
            tokenizer_class,
            config_class,
            model_path,
            tokenizer_path,
            config_only=config_only,
            hidden_dropout=hidden_dropout,
            attn_dropout=attn_dropout,
        )

        self.token_mask_prob = token_mask_prob
        self.embedding_dim = self.model.config.hidden_size
        self.fine_tune = fine_tune

        self.scalar_mix = self._init_scalar_mix(
            shared_embeddings=shared_embeddings, layer_dropout=scalar_mix_layer_dropout
        )
        self.word_piece_pooling = word_piece_pooling

        if output_dropout > 0.0:
            self.output_dropout = Dropout(p=output_dropout)
        else:
            self.output_dropout = None

    @classmethod
    def from_args_dict(cls, args_dict, model_dir=None):
        if model_dir is not None:
            args_dict["config_only"] = True
            args_dict["model_path"] = model_dir
            args_dict["tokenizer_path"] = model_dir / "tokenizer"

        return cls(**args_dict)

    def save_config(self, save_dir, prefix=""):
        """Save this module's transformer configuration to the specified directory."""
        makedirs(join(save_dir, prefix), exist_ok=True)

        self.model.config.to_json_file(join(save_dir, prefix, "config.json"))
        self.tokenizer.save_pretrained(join(save_dir, prefix, "tokenizer"))

    def _init_model(
        self,
        model_class,
        tokenizer_class,
        config_class,
        model_path,
        tokenizer_path,
        config_only=False,
        hidden_dropout=0.2,
        attn_dropout=0.2,
    ):
        """Initialize the transformer language model."""
        if config_only:
            model = model_class(config_class.from_json_file(str(model_path / "config.json")))
            tokenizer = tokenizer_class.from_pretrained(str(tokenizer_path))
        else:
            model = model_class.from_pretrained(
                model_path,
                output_hidden_states=True,
                hidden_dropout_prob=hidden_dropout,
                attention_probs_dropout_prob=attn_dropout,
                return_dict=True,
            )
            tokenizer = tokenizer_class.from_pretrained(model_path)

        return model, tokenizer

    def _init_scalar_mix(self, shared_embeddings=None, layer_dropout=0.1):
        """Initialize the scalar mixture module."""
        num_layers = self.model.config.num_hidden_layers + 1  # Add 1 because of input embeddings

        if shared_embeddings is None:
            scalar_mix = nn.ModuleDict(
                {
                    output_id: ScalarMixWithDropout(
                        mixture_size=num_layers, layer_dropout=layer_dropout
                    )
                    for output_id in self.output_ids
                }
            )
        else:
            scalar_mix = nn.ModuleDict()
            for group in shared_embeddings:
                curr_scalarmix = ScalarMixWithDropout(
                    mixture_size=num_layers, layer_dropout=layer_dropout
                )
                for outp_id in group:
                    scalar_mix[outp_id] = curr_scalarmix
            for outp_id in self.output_ids:
                if outp_id not in scalar_mix:
                    # Add scalar mixes for all outputs that don't have one yet
                    scalar_mix[outp_id] = ScalarMixWithDropout(
                        mixture_size=num_layers, layer_dropout=layer_dropout
                    )

        return scalar_mix

    def forward(self, input_sentences):
        """Transform a bunch of input sentences (list of lists of tokens) into a batch (tensor) of
        BERT/RoBERTa/etc. embeddings.

        Args:
            input_sentences: The input sentences to transform into embeddings (list of lists of tokens).

        Returns: A tuple consisting of (a) a dictionary with the embeddings for each output/annotation ID
          (shape: batch_size * max_seq_len * embedding_dim); (b) a tensor containing the length (number of tokens)
          of each sentence (shape: batch_size).
        """
        # Retrieve inputs for BERT model
        tokens, token_lengths, word_piece_ids, attention_mask = self._get_model_inputs(
            input_sentences
        )

        # Get embeddings tensors (= a dict containing one tensor for each output)
        raw_embeddings = self._get_raw_embeddings(word_piece_ids, attention_mask)

        # For each output, extract the token embeddings
        processed_embeddings = dict()
        for output_id in self.output_ids:
            processed_embeddings[output_id] = self._process_embeddings(
                raw_embeddings[output_id], tokens, token_lengths
            )

        # Sav true sequence lengths in a tensor
        true_seq_lengths = self._compute_true_seq_lengths(input_sentences, device=tokens.device)

        # If there is only one output, get rid of the dummy output ID
        if processed_embeddings.keys() == {"__dummy_output__"}:
            processed_embeddings = processed_embeddings["__dummy_output__"]

        return processed_embeddings, true_seq_lengths

    def _get_model_inputs(self, input_sentences):
        """Take a list of sentences and return tensors for token IDs, attention mask, and original token mask"""
        mask_prob = self.token_mask_prob if self.training else 0.0
        input_sequences = [
            TransformerInputSequence(sent, self.tokenizer, token_mask_prob=mask_prob)
            for sent in input_sentences
        ]
        device = next(iter(self.model.parameters())).device  # Ugly :(

        return TransformerInputSequence.batchify(input_sequences, device)

    def _get_raw_embeddings(self, word_piece_ids, attention_mask):
        """Take tensors for input tokens and run them through underlying BERT-based model, performing the learned scalar
        mixture for each output"""
        raw_embeddings = dict()

        with torch.set_grad_enabled(self.fine_tune):
            embedding_layers = torch.stack(
                self.model(word_piece_ids, attention_mask=attention_mask).hidden_states
            )

        for output_id in self.output_ids:
            if self.output_dropout:
                embedding_layers_with_dropout = self.output_dropout(embedding_layers)
                curr_output = self.scalar_mix[output_id](embedding_layers_with_dropout)
            else:
                curr_output = self.scalar_mix[output_id](embedding_layers)
            raw_embeddings[output_id] = curr_output

        return raw_embeddings

    def _process_embeddings(self, raw_embeddings, tokens, token_lengths):
        """Pool the raw word piece embeddings into token embeddings using the specified method."""
        batch_size = raw_embeddings.shape[0]
        embeddings_dim = raw_embeddings.shape[2]

        # Attach "neutral element" / padding to the raw embeddings tensor
        neutral_element = (
            -1e10 if self.word_piece_pooling == "max" else 0.0
        )  # Negative "infinity" for max pooling
        neutral_element_t = torch.empty(
            (1, 1, embeddings_dim), dtype=torch.float, device=raw_embeddings.device
        ).fill_(neutral_element)
        neutral_element_exp = neutral_element_t.expand((batch_size, 1, embeddings_dim))
        embeddings_with_neutral = torch.cat((raw_embeddings, neutral_element_exp), dim=1)

        # Gather the word piece embeddings corresponding to each token
        assert tokens.shape[0] == batch_size
        max_num_tokens = tokens.shape[1]
        max_wp_in_token = tokens.shape[2]
        tokens = (
            tokens.view(batch_size, max_num_tokens * max_wp_in_token)
            .unsqueeze(-1)
            .expand((-1, -1, embeddings_dim))
        )
        gathered_embeddings = torch.gather(embeddings_with_neutral, 1, tokens).view(
            batch_size, max_num_tokens, max_wp_in_token, embeddings_dim
        )

        # Pool values using the specified method
        if self.word_piece_pooling == "first":
            token_embeddings = gathered_embeddings[:, :, 0, :]
        elif self.word_piece_pooling == "sum":
            token_embeddings = torch.sum(gathered_embeddings, dim=2)
        elif self.word_piece_pooling == "avg":
            token_embeddings = torch.sum(gathered_embeddings, dim=2) / token_lengths.unsqueeze(-1)
        elif self.word_piece_pooling == "max":
            token_embeddings, _ = torch.max(gathered_embeddings, dim=2)
        else:
            raise Exception(f'Unknown pooling method "{self.word_piece_pooling}"!')

        return token_embeddings

    def _compute_true_seq_lengths(self, sentences, device=None):
        return torch.tensor([len(sent) for sent in sentences], device=device)

    def parallelize(self, device_ids):
        """Parallelize this module for multi-GPU setup-"""
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


class BertWrapper(TransformerWrapper):
    """Embeddings wrapper class for modules based on BERT."""

    def __init__(self, *args, **kwargs):
        super(BertWrapper, self).__init__(BertModel, BertTokenizer, BertConfig, *args, **kwargs)


class TransformerInputSequence:
    """Class for representing the features of a single, dependency-annotated sentence in tensor
    form, for usage in transformer-based models such as BERT.

    Example (BERT):
    ```
    Input sentence:                 Beware      the     jabberwock                 ,    my   son    !
    BERT word pieces:      [CLS]    be ##ware   the     ja ##bber   ##wo  ##ck     ,    my   son    ! [SEP]  ([PAD] [PAD] [PAD]  ...)
    BERT word piece IDs:     101  2022   8059  1996  14855  29325  12155  3600  1010  2026  2365  999   102  (    0     0     0  ...)
    BERT attention mask:       1     1      1     1      1      1      1     1     1     1     1    1     1  (    0     0     0  ...)

    Word-to-pieces mapping (non-padded): [[1,2], [3], [4,5,6,7], [8], [9], [10], [11]]
    ```
    """

    def __init__(self, orig_tokens, tokenizer, token_mask_prob=0.0):
        """
        Args:
            orig_tokens: Tokens to convert into a BertInputSequence.
            tokenizer: Tokenizer to use to split original tokens into word pieces.
            token_mask_prob: Probability of replacing an input token with a mask token. All word pieces of a given token
              will be replaced.
        """
        self.tokenizer = tokenizer

        self.word_pieces = list()
        self.attention_mask = list()
        self.tokens = list()
        self.token_lengths = list()

        self.append_special_token(self.tokenizer.cls_token)  # BOS marker

        ix = 1
        for orig_token in orig_tokens:
            tok_length = self.append_regular_token(orig_token, ix, mask_prob=token_mask_prob)
            ix += tok_length

        self.append_special_token(self.tokenizer.sep_token)  # EOS marker

        assert (
            len(orig_tokens)
            == len(self.tokens)
            <= len(self.word_pieces)
            == len(self.attention_mask)
        )

        # Convert word pieces to IDs
        self.word_piece_ids = self.tokenizer.convert_tokens_to_ids(self.word_pieces)

    def __len__(self):
        return len(self.word_pieces)

    def append_special_token(self, token):
        """Append a special token (e.g. BOS token, MASK token) to the sequence. The token will receive attention in the
        model, but will not be counted as an original token.
        """
        self.word_pieces.append(token)
        self.attention_mask.append(1)

    def append_regular_token(self, token, ix, mask_prob=0.0):
        """Append regular token (i.e., a word from the input sentence) to the sequence. The token will be split further
        into word pieces by the tokenizer."""
        curr_word_pieces = self.tokenizer.tokenize(token)

        if len(curr_word_pieces) == 0:
            curr_word_pieces = [self.tokenizer.unk_token]

        if token not in self.tokenizer.get_added_vocab():  # Do not mask special control tokens!
            if mask_prob > 0.0 and random.random() < mask_prob:
                curr_word_pieces = [self.tokenizer.mask_token] * len(curr_word_pieces)

        self.word_pieces += curr_word_pieces
        self.attention_mask += [1] * len(curr_word_pieces)
        self.tokens.append(list(range(ix, ix + len(curr_word_pieces))))
        self.token_lengths.append(len(curr_word_pieces))

        return len(curr_word_pieces)

    def pad_to_length(self, padded_num_tokens, padded_num_word_pieces, padded_max_wp_per_token):
        """Pad the sentence to the specified length. This will increase the length of all fields to padded_length by
        adding the padding label/index."""
        wp_padding_length = padded_num_word_pieces - len(self.word_pieces)
        seq_padding_length = padded_num_tokens - len(self.tokens)

        assert wp_padding_length >= 0
        assert seq_padding_length >= 0
        assert padded_max_wp_per_token >= max(len(token) for token in self.tokens)

        self.word_pieces += [self.tokenizer.pad_token] * wp_padding_length
        self.word_piece_ids += [self.tokenizer.pad_token_id] * wp_padding_length
        self.attention_mask += [0] * wp_padding_length

        wp_padding_ix = padded_num_word_pieces  # noqa: F841
        for i, curr_token in enumerate(self.tokens):
            curr_padding_length = padded_max_wp_per_token - len(curr_token)
            self.tokens[i] = curr_token + [padded_num_word_pieces] * curr_padding_length

        self.tokens += [
            [padded_num_word_pieces] * padded_max_wp_per_token for _ in range(seq_padding_length)
        ]
        self.token_lengths += [1] * seq_padding_length

        assert len(self.word_pieces) == len(self.word_piece_ids) == len(self.attention_mask)
        assert all(len(tok) == len(self.tokens[0]) for tok in self.tokens)

    def tensorize(
        self,
        device,
        padded_num_tokens=None,
        padded_num_word_pieces=None,
        padded_max_wp_per_token=None,
    ):
        if len(self.word_piece_ids) > 512:
            self._throw_out_non_first_word_pieces()
            assert len(self.word_piece_ids) < 512

        if padded_num_tokens is None:
            padded_num_tokens = len(self.tokens)
        if padded_num_word_pieces is None:
            padded_num_word_pieces = len(self.word_pieces)
        if padded_max_wp_per_token is None:
            padded_max_wp_per_token = max(len(token) for token in self.tokens)

        self.pad_to_length(padded_num_tokens, padded_num_word_pieces, padded_max_wp_per_token)

        self.word_piece_ids = torch.tensor(self.word_piece_ids, device=device)
        self.attention_mask = torch.tensor(self.attention_mask, device=device)
        self.tokens = torch.tensor(self.tokens, device=device)
        self.token_lengths = torch.tensor(self.token_lengths, device=device)

    def _throw_out_non_first_word_pieces(self):
        self.word_pieces = (
            [self.tokenizer.cls_token]
            + [self.word_pieces[tok[0]] for tok in self.tokens]
            + [self.tokenizer.sep_token]
        )
        self.word_piece_ids = (
            [self.tokenizer.cls_token_id]
            + [self.word_piece_ids[tok[0]] for tok in self.tokens]
            + [self.tokenizer.sep_token_id]
        )
        self.attention_mask = [1] * (len(self.tokens) + 2)

        self.tokens = [[i + 1] for i in range(len(self.tokens))]
        self.token_lengths = [1] * len(self.tokens)

    @staticmethod
    def batchify(input_seqs, device):
        padded_num_tokens = max(len(input_seq.tokens) for input_seq in input_seqs)
        padded_num_word_pieces = min(
            max(len(input_seq.word_pieces) for input_seq in input_seqs), 512
        )
        padded_max_wp_per_token = max(
            len(token) for input_seq in input_seqs for token in input_seq.tokens
        )

        for input_seq in input_seqs:
            input_seq.tensorize(
                device,
                padded_num_tokens=padded_num_tokens,
                padded_num_word_pieces=padded_num_word_pieces,
                padded_max_wp_per_token=padded_max_wp_per_token,
            )

        tokens = torch.stack([input_seq.tokens for input_seq in input_seqs])
        token_lengths = torch.stack([input_seq.token_lengths for input_seq in input_seqs])
        word_piece_ids = torch.stack([input_seq.word_piece_ids for input_seq in input_seqs])
        attention_mask = torch.stack([input_seq.attention_mask for input_seq in input_seqs])

        return tokens, token_lengths, word_piece_ids, attention_mask

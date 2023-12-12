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
This module contains the multi-task BERT+CRF classifier used for multi-task NER experiments.
"""

from typing import Iterator

import torch
from torch import nn
from transformers import BertModel

from source.ner.modules.crf import CRF


class MT_NER_CRF_Classifier(nn.Module):
    """
    The NER multi-task model. It uses one shared BERT LM and multiple linear layers + CRF layers to produce predictions; one for each distict dataset.
    """

    def __init__(self, bert_model_name: str, list_of_labels: list[list[str]]) -> None:
        """
        Creates a new instance of a multi-task BERT+CRF classifier.

        Args:
            bert_model_name (str): Path/Name of BERT LM model
            list_of_labels (list[list[str]]): Batch of "Label lists" -> use correct ordering s.t. training batches are later matched to the correct layers
        """
        super().__init__()
        self._bert_model: BertModel = BertModel.from_pretrained(bert_model_name)
        self._dropout: nn.Dropout = nn.Dropout(p=0.2)
        self._linear_layers: nn.ModuleList = nn.ModuleList()
        self._crf_layers: nn.ModuleList = nn.ModuleList()
        for label_list in list_of_labels:
            self._linear_layers.append(
                nn.Linear(
                    in_features=self._bert_model.config.hidden_size, out_features=len(label_list)
                )
            )
            self._crf_layers.append(CRF(num_tags=len(label_list), batch_first=True))

    def get_lm_parameters(self) -> Iterator[torch.Tensor]:
        """
        Returns the trainable parameters of the BERT LM and linear output layers.

        Yields:
            Iterator[torch.Tensor]: Iterator of trainable model layers
        """
        layers: list = [self._bert_model.parameters()] + [
            linear_layer.parameters() for linear_layer in self._linear_layers
        ]
        for lay in layers:
            for p in lay:
                yield p

    def get_crf_parameters(self) -> Iterator[torch.Tensor]:
        """
        Returns the trainable parameters of the CRF layers.

        Yields:
            Iterator[torch.Tensor]: Iterator of trainable CRF parameters
        """
        for crf in self._crf_layers:
            for p in crf.parameters():
                yield p

    def get_linear_layer_parameters_by_index(self, idx: int) -> Iterator[torch.Tensor]:
        """
        Allows to index the trainable parameters of a specific linear output layer.

        Args:
            idx (int): Index to the desired linear layer. Must be within [0; num_datasets - 1]

        Yields:
            Iterator[torch.Tensor]: Iterator of trainable parameters of indexed layer
        """
        for p in self._linear_layers[idx].parameters():
            yield p

    def get_crf_layer_parameters_by_index(self, idx: int) -> Iterator[torch.Tensor]:
        """
        Allows to index the trainable parameters of a specific CRF layer.

        Args:
            idx (int): Index to the desired CRF layer. Must be within [0; num_datasets - 1]

        Yields:
            Iterator[torch.Tensor]: Iterator of trainable parameters of indexed CRF
        """
        for p in self._crf_layers[idx].parameters():
            yield p

    def get_lm_parameters_only(self) -> Iterator[torch.Tensor]:
        """
        Returns only the parameters of the BERT language model.

        Yields:
            Iterator[torch.Tensor]: Iterator of trainable BERT layers
        """
        for p in self._bert_model.parameters():
            yield p

    def _get_linear_output_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        dataset_flag: int,
    ) -> torch.Tensor:
        """
        Returns the logits produced by BERT and the linear output layer on top. The linear layer is selected based on the dataset from which the current input sample originates.

        Args:
            input_ids (torch.Tensor): BERT input IDs
            attention_mask (torch.Tensor): BERT attention mask
            token_type_ids (torch.Tensor): BERT token type IDs
            dataset_flag (int): Indicates the dataset from which the current sample comes from

        Returns:
            torch.Tensor: Per-token logits
        """
        embeddings: torch.Tensor = self._bert_model(
            input_ids, attention_mask, token_type_ids
        ).last_hidden_state
        embeddings = self._dropout(embeddings)
        logits: torch.Tensor = self._linear_layers[dataset_flag](embeddings)
        return logits

    def get_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        crf_mask: torch.Tensor,
        sorted_crf_mask: torch.Tensor,
        tag_sequence: torch.Tensor,
        dataset_flag: int,
    ) -> torch.Tensor:
        """
        Returns the loss between prediction from the CRF layer and ground truth sequence. It first uses the _get_linear_output_logits() function to produce the logits for the CRF layer.
        The linear and CRF layers are selected based on the dataset from which the current input sample originates.

        Args:
            input_ids (torch.Tensor): BERT input IDs
            attention_mask (torch.Tensor): BERT attention mask
            token_type_ids (torch.Tensor): BERT token type IDs
            crf_mask (torch.Tensor): CRF mask
            sorted_crf_mask (torch.Tensor): Sorted variant of the CRF mask
            tag_sequence (torch.Tensor): Ground truth tag sequence
            dataset_flag (int): Indicates the dataset from which the current sample comes from

        Returns:
            torch.Tensor: The loss calculated by the CRF module
        """
        logits: torch.Tensor = self._get_linear_output_logits(
            input_ids, attention_mask, token_type_ids, dataset_flag
        )
        pad_size: int = logits.shape[-2]
        seq_len: int = logits.shape[-1]
        sorted_logits: list[torch.Tensor] = [
            torch.stack([score for j, score in enumerate(l) if crf_mask[i][j]])
            for i, l in enumerate(logits)
        ]
        for i in range(len(sorted_logits)):
            for _ in range(len(sorted_logits[i]), pad_size):
                if len(sorted_logits[i]) < pad_size:
                    sorted_logits[i] = torch.cat(
                        [
                            sorted_logits[i],
                            torch.stack(
                                [
                                    torch.tensor([0] * seq_len).to(logits.device)
                                    for _ in range(len(sorted_logits[i]), pad_size)
                                ]
                            ),
                        ]
                    )
        return self._crf_layers[dataset_flag](
            torch.stack(sorted_logits), tag_sequence, sorted_crf_mask, reduction="sum"
        )

    def predict_tag_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        crf_mask: torch.Tensor,
        sorted_crf_mask: torch.Tensor,
        dataset_flag: int,
    ) -> list[list[int]]:
        """
        Returns the BILOU sequence preditions produced by this model. The linear and CRF layers are selected based on the dataset from which the current input sample originates.

        Args:
            input_ids (torch.Tensor): BERT input IDs
            attention_mask (torch.Tensor): BERT attention mask
            token_type_ids (torch.Tensor): BERT token type IDs
            crf_mask (torch.Tensor): CRF mask
            sorted_crf_mask (torch.Tensor): Sorted CRF mask
            dataset_flag (int): Indicates the dataset from which the current sample comes from

        Returns:
            list[list[int]]: Batch of predicted BILOU labels ()
        """
        logits: torch.Tensor = self._get_linear_output_logits(
            input_ids, attention_mask, token_type_ids, dataset_flag
        )
        pad_size: int = logits.shape[-2]
        seq_len: int = logits.shape[-1]
        sorted_logits: list[torch.Tensor] = [
            torch.stack([score for j, score in enumerate(l) if crf_mask[i][j]])
            for i, l in enumerate(logits)
        ]
        for i in range(len(sorted_logits)):
            for _ in range(len(sorted_logits[i]), pad_size):
                if len(sorted_logits[i]) < pad_size:
                    sorted_logits[i] = torch.cat(
                        [
                            sorted_logits[i],
                            torch.stack(
                                [
                                    torch.tensor([0] * seq_len).to(logits.device)
                                    for _ in range(len(sorted_logits[i]), pad_size)
                                ]
                            ),
                        ]
                    )
        return self._crf_layers[dataset_flag].decode(torch.stack(sorted_logits), sorted_crf_mask)

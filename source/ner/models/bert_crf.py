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
This module contains the BERT+CRF-based NER classifier.
"""

from typing import Iterator

import torch
from torch import nn
from transformers import BertModel

from source.ner.modules.crf import CRF


class NER_CRF_Classifier(nn.Module):
    """
    A BERT-based classifier used for NER that feeds predicted logits into a conditional random field (CRF) to produce consistent BILOU tagging sequences.
    """

    def __init__(self, bert_model_name: str, labels: list[str]) -> None:
        """
        Initializes the classifier.

        Args:
            bert_model_name (str): Path or name of BERT model
            labels (list[str]): Available BILOU labels
        """
        super().__init__()
        self._bert_model: BertModel = BertModel.from_pretrained(bert_model_name)
        self._dropout: nn.Dropout = nn.Dropout(p=0.2)
        self._linear: nn.Linear = nn.Linear(
            in_features=self._bert_model.config.hidden_size, out_features=len(labels)
        )
        self._crf: CRF = CRF(num_tags=len(labels), batch_first=True)

    def get_lm_parameters(self) -> Iterator[torch.Tensor]:
        """
        Returns the trainable parameters of the BERT LM and linear output layer.

        Yields:
            Iterator[torch.Tensor]: Iterator of trainable model layers
        """
        layers: list = [self._bert_model.parameters(), self._linear.parameters()]
        for lay in layers:
            for p in lay:
                yield p

    def get_crf_parameters(self) -> Iterator[torch.Tensor]:
        """
        Returns the trainable parameters of the CRF layer.

        Yields:
            Iterator[torch.Tensor]: Iterator of trainable CRF parameters
        """
        for p in self._crf.parameters():
            yield p

    def _get_linear_output_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the logits produced by BERT and the linear output layer on top.

        Args:
            input_ids (torch.Tensor): BERT input IDs
            attention_mask (torch.Tensor): BERT attention mask
            token_type_ids (torch.Tensor): BERT token type IDs

        Returns:
            torch.Tensor: Per-token logits
        """
        embeddings: torch.Tensor = self._bert_model(
            input_ids, attention_mask, token_type_ids
        ).last_hidden_state
        embeddings = self._dropout(embeddings)
        logits: torch.Tensor = self._linear(embeddings)
        return logits

    def get_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        crf_mask: torch.Tensor,
        sorted_crf_mask: torch.Tensor,
        tag_sequence: torch.Tensor,
        unused_dataset_flag: int = None,
    ) -> torch.Tensor:
        """
        Returns the loss between prediction from the CRF layer and ground truth sequence. It first uses the _get_linear_output_logits() function to produce the logits for the CRF layer.

        Args:
            input_ids (torch.Tensor): BERT input IDs
            attention_mask (torch.Tensor): BERT attention mask
            token_type_ids (torch.Tensor): BERT token type IDs
            crf_mask (torch.Tensor): CRF mask
            sorted_crf_mask (torch.Tensor): Sorted variant of the CRF mask
            tag_sequence (torch.Tensor): Ground truth tag sequence
            unused_dataset_flag (int, optional): Only used in a multi-task scenario. Defaults to None.

        Returns:
            torch.Tensor: The loss calculated by the CRF module
        """
        logits: torch.Tensor = self._get_linear_output_logits(
            input_ids, attention_mask, token_type_ids
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
        return self._crf(
            torch.stack(sorted_logits), tag_sequence, sorted_crf_mask, reduction="sum"
        )

    def predict_tag_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        crf_mask: torch.Tensor,
        sorted_crf_mask: torch.Tensor,
        unused_dataset_flag: int = None,
    ) -> list[list[int]]:
        """
        Returns the BILOU sequence preditions produced by this model.

        Args:
            input_ids (torch.Tensor): BERT input IDs
            attention_mask (torch.Tensor): BERT attention mask
            token_type_ids (torch.Tensor): BERT token type IDs
            crf_mask (torch.Tensor): CRF mask
            sorted_crf_mask (torch.Tensor): Sorted CRF mask
            unused_dataset_flag (int, optional): Only used in a multi-task scenario. Defaults to None.

        Returns:
            list[list[int]]: Batch of predicted BILOU sequences
        """
        logits: torch.Tensor = self._get_linear_output_logits(
            input_ids, attention_mask, token_type_ids
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
        return self._crf.decode(torch.stack(sorted_logits), sorted_crf_mask)

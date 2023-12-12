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
This module contains the BERT-based measurement classifier model and the random classifier baseline.
"""

import random

import torch
from torch import nn
from transformers import BertConfig, BertModel, PretrainedConfig

from source.constants.mulms_constants import meas_labels
from source.measurement_classification.datasets.measurement_dataset import (
    MeasurementDataset,
)


class MeasurementClassifier(nn.Module):
    """
    The BERT-based measurement classifier.
    """

    def __init__(self, model_path: str, num_labels: int, dropout_rate: float = 0.1) -> None:
        """
        Initializes the classifier.

        Args:
            model_path (str): Path or name of BERT-based model.
            num_labels (int): Number of labels of the multi-class problem.
            dropout_rate (float, optional): Dropout rate for the layer between BERT and Linear. Defaults to 0.1.
        """
        super().__init__()
        self.encoder: BertModel = BertModel.from_pretrained(model_path)
        config: PretrainedConfig = BertConfig.from_pretrained(model_path)
        self.cls_size: int = int(config.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(p=dropout_rate)
        self.linear_layer: nn.Linear = nn.Linear(self.cls_size, num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the classifier

        Args:
            input_ids (torch.Tensor): Input IDs as returned by a tokenizer
            attention_mask (torch.Tensor): Attention mask values
            token_type_ids (torch.Tensor): Token Type IDs

        Returns:
            torch.Tensor: Logits
        """
        model_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        encoded_cls: torch.Tensor = model_outputs.last_hidden_state[:, 0]
        encoded_cls_dp: torch.Tensor = self.dropout(encoded_cls)
        logits: torch.Tensor = self.linear_layer(encoded_cls_dp)
        return logits


class RandomMeasurementClassifier:
    """
    The random baseline classifier. It computes a-priori probabilites based on label counts in the MuLMS dataset and applies random classification using these probabilities.
    """

    def __init__(self, prior_dataset: MeasurementDataset) -> None:
        """
        Initializes the random classifier by creating the a-priori likelihoods.

        Args:
            prior_dataset (MeasurementDataset): The MuLMS dataset to compute the a-priori scores on.
        """
        self.prior_probabilities: dict = {c: {"P": 0, "count": 0} for c in meas_labels}

        for label in prior_dataset._data["labels"]:
            self.prior_probabilities[label]["count"] += 1

        total_count: int = len(prior_dataset._data["labels"])

        for _, v in self.prior_probabilities.items():
            v["P"] = v["count"] / float(total_count)

        assert (
            abs(sum([v["P"] for _, v in self.prior_probabilities.items()]) - 1.0) < 1e-6
        ), "Prior probabilities don't add up to one!"

    def predict(self, dataset: MeasurementDataset) -> list[str]:
        """
        Randomly classifies the samples.

        Args:
            dataset (MeasurementDataset): MuLMS dataset to classify.

        Returns:
            list[str]: List of predictions; one entry for each sample.
        """
        choices: list[str] = []
        prob_weights: list[float] = []
        for k, v in self.prior_probabilities.items():
            choices.append(k)
            prob_weights.append(v["P"])
        return random.choices(choices, weights=prob_weights, k=len(dataset._data["labels"]))

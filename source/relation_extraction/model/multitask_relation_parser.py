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
from pathlib import Path

import torch
from torch import nn

from source.relation_extraction.model.relation_parser import MuLMSRelationParser


class MultitaskRelationParser(nn.Module):
    def __init__(self, relation_parsers):
        super().__init__()

        # Register individual parsers
        assert all(isinstance(rel_parser, MuLMSRelationParser) for rel_parser in relation_parsers)
        self.relation_parsers = nn.ModuleList(relation_parsers)

        # Share language model across parsers
        for rel_parser in self.relation_parsers:
            rel_parser.embed = self.relation_parsers[0].embed

    @classmethod
    def from_folder(cls, model_folder):
        model_folder = Path(model_folder)
        with open(model_folder / "config.json", "r") as config_json_file:
            model_config = json.load(config_json_file)["model"]

        # Create individual relation parsers
        relation_parsers = list()
        for i, rel_parser_config in enumerate(model_config["relation_parsers"]):
            rel_parser = MuLMSRelationParser.from_folder(model_folder / f"parser{i}")
            relation_parsers.append(rel_parser)

        # Create model
        model = cls(relation_parsers)

        # Load saved weights
        checkpoint = torch.load(model_folder / "model_best.pth", map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

        return model

    def save_config(self, save_dir, prefix=""):
        for i, rel_parser in enumerate(self.relation_parsers):
            rel_parser.save_config(save_dir, prefix=prefix + f"/parser{i}")

    def forward(
        self,
        data_source_index,
        input_sents,
        num_ne,
        ne_pos,
        ne_labels,
        mode="evaluation",
        targets=None,
        post_process=False,
    ):
        return self.relation_parsers[data_source_index](
            input_sents,
            num_ne,
            ne_pos,
            ne_labels,
            mode=mode,
            targets=targets,
            post_process=post_process,
        )

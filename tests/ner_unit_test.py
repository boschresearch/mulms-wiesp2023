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
This module contains unit tests for the NER pipeline.
"""

import os
import pathlib
import unittest

from source.constants.mulms_constants import mulms_ne_labels
from source.ner.evaluation.metrics import (
    calculate_prec_recall_f1_for_bilou,
    calculate_prec_recall_f1_for_dep_parsing,
)

data_input_path: str = os.path.join(
    str(pathlib.Path(__file__).parent.resolve()), "../data/ml_ms_corpus"
)
data_input_files: list = [
    os.path.join(data_input_path, "xmi", f)
    for f in os.listdir(os.path.join(data_input_path, "xmi"))
    if "TypeSystem" not in f
]


class EvaluationTester(unittest.TestCase):
    """
    The unit testing class.
    """

    def test_ner_dep_parsing_score_calculation(self):
        """
        Tests the metrics computation function for the NER as dependency parsing module.
        """
        prediction: list[list[list[str]]] = [
            [
                ["O", "O", "MAT", "NUM"],
                ["O", "VALUE+NUM", "O", "O"],
                ["CITE", "O", "TECHNIQUE", "O"],
                ["O", "O", "O", "O"],
            ],
            [
                ["O", "O", "MAT+FORM", "FORM"],
                ["O", "SAMPLE", "O", "O"],
                ["CITE", "CITE", "TECHNIQUE", "O"],
                ["O", "RANGE", "O", "INSTRUMENT"],
            ],
        ]

        gt: list[set[tuple]] = [
            set(
                [
                    tuple(["MAT", 0, 2]),
                    tuple(["NUM", 1, 1]),
                    tuple(["CITE", 2, 0]),
                    tuple(["TECHNIQUE", 2, 2]),
                    tuple(["VALUE", 3, 0]),
                ]
            ),
            set(
                [
                    tuple(["MAT", 0, 2]),
                    tuple(["FORM", 0, 2]),
                    tuple(["FORM", 0, 3]),
                    tuple(["RANGE", 1, 1]),
                    tuple(["UNIT", 1, 3]),
                    tuple(["CITE", 2, 0]),
                    tuple(["MAT", 2, 1]),
                    tuple(["FORM", 2, 1]),
                    tuple(["TECHNIQUE", 2, 2]),
                    tuple(["RANGE", 3, 1]),
                    tuple(["INSTRUMENT", 3, 3]),
                ]
            ),
        ]

        scores: dict = calculate_prec_recall_f1_for_dep_parsing(
            prediction=prediction, gold_labels=gt
        )

        self.assertAlmostEqual(scores["MAT"]["P"], 1)
        self.assertAlmostEqual(scores["MAT"]["R"], 2.0 / 3)
        self.assertAlmostEqual(scores["MAT"]["F1"], 0.8)

        self.assertAlmostEqual(scores["CITE"]["P"], 2.0 / 3)
        self.assertAlmostEqual(scores["CITE"]["R"], 1)
        self.assertAlmostEqual(scores["CITE"]["F1"], 0.8)

        self.assertAlmostEqual(scores["TECHNIQUE"]["P"], 1)
        self.assertAlmostEqual(scores["TECHNIQUE"]["R"], 1)
        self.assertAlmostEqual(scores["TECHNIQUE"]["F1"], 1)

        self.assertAlmostEqual(scores["RANGE"]["P"], 1)
        self.assertAlmostEqual(scores["RANGE"]["R"], 0.5)
        self.assertAlmostEqual(scores["RANGE"]["F1"], 2.0 / 3)

        self.assertAlmostEqual(scores["VALUE"]["P"], 0)
        self.assertAlmostEqual(scores["VALUE"]["R"], 0)
        self.assertAlmostEqual(scores["VALUE"]["F1"], 0)

        self.assertAlmostEqual(scores["UNIT"]["P"], 0)
        self.assertAlmostEqual(scores["UNIT"]["R"], 0)
        self.assertAlmostEqual(scores["UNIT"]["F1"], 0)

        self.assertAlmostEqual(scores["NUM"]["P"], 0.5)
        self.assertAlmostEqual(scores["NUM"]["R"], 1)
        self.assertAlmostEqual(scores["NUM"]["F1"], 2.0 / 3)

        self.assertAlmostEqual(scores["FORM"]["P"], 1)
        self.assertAlmostEqual(scores["FORM"]["R"], 2.0 / 3)
        self.assertAlmostEqual(scores["FORM"]["F1"], 0.8)

        self.assertAlmostEqual(scores["INSTRUMENT"]["P"], 1)
        self.assertAlmostEqual(scores["INSTRUMENT"]["R"], 1)
        self.assertAlmostEqual(scores["INSTRUMENT"]["F1"], 1)

        self.assertAlmostEqual(
            scores["Macro_F1"],
            (0.8 + 0.8 + 1 + 2.0 / 3 + 2.0 / 3 + 0.8 + 1) / len(mulms_ne_labels),
        )
        self.assertAlmostEqual(scores["Micro_F1"], 22.0 / 31)

    def test_ner_bilou_score_calculation(self):
        """
        Tests the metrics computation function for the NER BILOU module.
        """
        prediction: list[list[str]] = [
            [
                "O",
                "B-CITE",
                "L-CITE",
                "O",
                "B-DEV",
                "I-DEV",
                "I-DEV+U-MAT",
                "L-DEV",
                "O",
                "U-NUM",
                "U-MAT",
            ],
            [
                "B-RANGE+B-VALUE",
                "I-RANGE+I-VALUE+U-UNIT",
                "L-RANGE+L-VALUE+U-NUM",
                "O",
                "B-CITE",
                "I-CITE",
                "L-CITE",
            ],
            [
                "O",
                "U-PROPERTY+U-MEASUREMENT",
                "U-VALUE+U-UNIT",
                "B-RANGE+B-VALUE",
                "I-RANGE+B-VALUE",
                "L-RANGE",
                "O",
                "I-MAT",
                "U-VALUE",
                "U-NUM",
                "U-NUM",
            ],
        ]
        gt: list[set[tuple]] = [
            set(
                [
                    tuple(["MAT", 0, 0]),
                    tuple(["CITE", 1, 2]),
                    tuple(["DEV", 4, 7]),
                    tuple(["MAT", 6, 6]),
                ]
            ),
            set(
                [
                    tuple(["RANGE", 0, 2]),
                    tuple(["CITE", 4, 6]),
                    tuple(["VALUE", 0, 2]),
                    tuple(["UNIT", 1, 1]),
                    tuple(["NUM", 2, 2]),
                    tuple(["SAMPLE", 3, 3]),
                ]
            ),
            set(
                [
                    tuple(["PROPERTY", 1, 1]),
                    tuple(["VALUE", 2, 2]),
                    tuple(["UNIT", 2, 2]),
                    tuple(["RANGE", 3, 5]),
                ]
            ),
        ]

        scores: dict = calculate_prec_recall_f1_for_bilou(
            predictions=prediction, gold_labels=gt, ne_labels=mulms_ne_labels
        )

        self.assertAlmostEqual(scores["MAT"]["P"], 0.5)
        self.assertAlmostEqual(scores["MAT"]["R"], 0.5)
        self.assertAlmostEqual(scores["MAT"]["F1"], 0.5)

        self.assertAlmostEqual(scores["CITE"]["P"], 1)
        self.assertAlmostEqual(scores["CITE"]["R"], 1)
        self.assertAlmostEqual(scores["CITE"]["F1"], 1)

        self.assertAlmostEqual(scores["DEV"]["P"], 1)
        self.assertAlmostEqual(scores["DEV"]["R"], 1)
        self.assertAlmostEqual(scores["DEV"]["F1"], 1)

        self.assertAlmostEqual(scores["RANGE"]["P"], 1)
        self.assertAlmostEqual(scores["RANGE"]["R"], 1)
        self.assertAlmostEqual(scores["RANGE"]["F1"], 1)

        self.assertAlmostEqual(scores["VALUE"]["P"], 2.0 / 3.0)
        self.assertAlmostEqual(scores["VALUE"]["R"], 1)
        self.assertAlmostEqual(scores["VALUE"]["F1"], 0.8)

        self.assertAlmostEqual(scores["UNIT"]["P"], 1)
        self.assertAlmostEqual(scores["UNIT"]["R"], 1)
        self.assertAlmostEqual(scores["UNIT"]["F1"], 1)

        self.assertAlmostEqual(scores["NUM"]["P"], 1.0 / 4.0)
        self.assertAlmostEqual(scores["NUM"]["R"], 1)
        self.assertAlmostEqual(scores["NUM"]["F1"], 0.4)

        self.assertAlmostEqual(scores["SAMPLE"]["P"], 0)
        self.assertAlmostEqual(scores["SAMPLE"]["R"], 0)
        self.assertAlmostEqual(scores["SAMPLE"]["F1"], 0)

        self.assertAlmostEqual(scores["PROPERTY"]["P"], 1)
        self.assertAlmostEqual(scores["PROPERTY"]["R"], 1)
        self.assertAlmostEqual(scores["PROPERTY"]["F1"], 1)

        self.assertAlmostEqual(
            scores["Macro_F1"],
            (0.5 + 1 + 1 + 1 + 0.8 + 1 + 0.4 + 0 + 1) / len(mulms_ne_labels),
        )
        self.assertAlmostEqual(scores["Micro_F1"], 0.75)

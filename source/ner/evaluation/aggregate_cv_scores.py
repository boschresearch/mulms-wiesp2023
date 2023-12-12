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
Calculates all metric scores across n (default: 5) folds.
"""

import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from source.constants.mulms_constants import (
    mspt_ne_labels,
    mulms_ne_labels,
    sofc_ne_labels,
)

parser: ArgumentParser = ArgumentParser()
parser.add_argument(
    "--input_path", type=str, help="Path to directoy which all CV scores are stored in."
)
parser.add_argument("--num_folds", type=int, help="Only change if really necessary.", default=5)
parser.add_argument(
    "--set",
    type=str,
    choices=["dev", "test"],
    help="Determines which split to evaluate.",
    default="test",
)
parser.add_argument(
    "--export_as_latex_table", action="store_true", help="Whether to export scores as Latex table."
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["MULMS", "SOFC", "MSPT"],
    help="Determines which NER dataset to evaluate.",
    default="MULMS",
)

args = parser.parse_args()


def main():
    """
    Entry point.
    """

    ne_labels: list[str] = None
    if args.dataset == "MULMS":
        ne_labels = mulms_ne_labels
    elif args.dataset == "SOFC":
        ne_labels = sofc_ne_labels
    elif args.dataset == "MSPT":
        ne_labels = mspt_ne_labels

    result_scores: list[dict] = []
    for fold in range(1, args.num_folds + 1):
        with open(
            os.path.join(args.input_path, f"cv_{fold}", f"scores_{args.set}.pickle"), mode="rb"
        ) as f:
            result_scores.append(pickle.load(f))
    micro_f1_scores: list[float] = [r["Micro_F1"] for r in result_scores]
    macro_f1_scores: list[float] = [r["Macro_F1"] for r in result_scores]
    average_scores: dict = dict([(label, {"P": 0.0, "R": 0.0, "F1": 0.0}) for label in ne_labels])
    average_scores["Micro_F1"] = np.average(micro_f1_scores)
    average_scores["Macro_F1"] = np.average(macro_f1_scores)
    for ne_label in ne_labels:
        average_scores[ne_label]["P"] = np.average([r[ne_label]["P"] for r in result_scores])
        average_scores[ne_label]["R"] = np.average([r[ne_label]["R"] for r in result_scores])
        average_scores[ne_label]["F1"] = np.average([r[ne_label]["F1"] for r in result_scores])
        average_scores[ne_label]["P_std"] = np.std([r[ne_label]["P"] for r in result_scores])
        average_scores[ne_label]["R_std"] = np.std([r[ne_label]["R"] for r in result_scores])
        average_scores[ne_label]["F1_std"] = np.std([r[ne_label]["F1"] for r in result_scores])

    std_dev_micro_f1: float = np.std(micro_f1_scores)
    std_dev_macro_f1: float = np.std(macro_f1_scores)

    print(f"Average Micro F1: {round(100 * average_scores['Micro_F1'], 1)}")
    print(f"Std. Dev. of Micro F1: {round(100 * std_dev_micro_f1, 1)}")
    print(f"Average Macro F1: {round(100 * average_scores['Macro_F1'], 1)}")
    print(f"Std. Dev. of Macro F1: {round(100 * std_dev_macro_f1, 1)}")

    average_scores.pop("Micro_F1")
    average_scores.pop("Macro_F1")
    for k in average_scores.keys():
        print(
            f"{k}: Precision: {round(100 * average_scores[k]['P'], 1)}, Recall: {round(100 * average_scores[k]['R'], 1)}, F1: {round(100 * average_scores[k]['F1'], 1)}"
        )

    if args.export_as_latex_table:
        df: pd.DataFrame = (
            pd.DataFrame.from_dict(average_scores).transpose().apply(lambda x: round(100 * x, 1))
        )
        df.to_latex(os.path.join(args.input_path, f"scores_{args.set}.tex"))

    return


if __name__ == "__main__":
    main()

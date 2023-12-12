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

from source.constants.mulms_constants import meas_labels

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
    "-add_class_none_to_avg_scores",
    action="store_true",
    help="If true, average scores will incorporate results of class None.",
)

args = parser.parse_args()


def main():
    """
    Entry point.
    """
    result_scores: list[dict] = []
    for fold in range(1, args.num_folds + 1):
        with open(
            os.path.join(args.input_path, f"cv_{fold}", f"scores_{args.set}.pickle"), mode="rb"
        ) as f:
            result_scores.append(pickle.load(f))

    average_scores: dict = dict(
        [(label, {"P": 0.0, "R": 0.0, "F1": 0.0}) for label in meas_labels]
    )

    for meas_label in meas_labels:
        average_scores[meas_label]["P"] = np.average([r[meas_label]["P"] for r in result_scores])
        average_scores[meas_label]["R"] = np.average([r[meas_label]["R"] for r in result_scores])
        average_scores[meas_label]["F1"] = np.average([r[meas_label]["F"] for r in result_scores])
        average_scores[meas_label]["P_std"] = np.std([r[meas_label]["P"] for r in result_scores])
        average_scores[meas_label]["R_std"] = np.std([r[meas_label]["R"] for r in result_scores])
        average_scores[meas_label]["F1_std"] = np.std([r[meas_label]["F"] for r in result_scores])

    if args.add_class_none_to_avg_scores:

        micro_f1_scores: list[float] = [r["Micro_F1"] for r in result_scores]
        macro_f1_scores: list[float] = [r["Macro_F1"] for r in result_scores]
        micro_p_scores: list[float] = [r["Micro_P"] for r in result_scores]
        macro_p_scores: list[float] = [r["Macro_P"] for r in result_scores]
        micro_r_scores: list[float] = [r["Micro_R"] for r in result_scores]
        macro_r_scores: list[float] = [r["Macro_R"] for r in result_scores]

        average_scores["Micro_F1"] = np.average(micro_f1_scores)
        average_scores["Macro_F1"] = np.average(macro_f1_scores)
        average_scores["Micro_P"] = np.average(micro_p_scores)
        average_scores["Macro_P"] = np.average(macro_p_scores)
        average_scores["Micro_R"] = np.average(micro_r_scores)
        average_scores["Macro_R"] = np.average(macro_r_scores)

        std_dev_micro_f1: float = round(100 * np.std(micro_f1_scores), 1)
        std_dev_macro_f1: float = round(100 * np.std(macro_f1_scores), 1)
        std_dev_micro_p: float = round(100 * np.std(micro_p_scores), 1)
        std_dev_macro_p: float = round(100 * np.std(macro_p_scores), 1)
        std_dev_micro_r: float = round(100 * np.std(micro_r_scores), 1)
        std_dev_macro_r: float = round(100 * np.std(macro_r_scores), 1)

        print(f"Average Micro F1: {round(100 * average_scores['Micro_F1'], 1)}")
        print(f"Std. Dev. of Micro F1: {std_dev_micro_f1}")
        print(f"Average Macro F1: {round(100 * average_scores['Macro_F1'], 1)}")
        print(f"Std. Dev. of Macro F1: {std_dev_macro_f1}")
        print(
            f"Macro P: {round(100 * average_scores['Macro_P'], 1)}; Std. Macro P: {std_dev_macro_p}; Micro P: {round(100 * average_scores['Micro_P'], 1)}; Std. Micro P: {std_dev_micro_p}"
        )
        print(
            f"Macro R: {round(100 * average_scores['Macro_R'], 1)}; Std. Macro R: {std_dev_macro_r}; Micro R: {round(100 * average_scores['Micro_R'], 1)}; Std. Micro R: {std_dev_micro_r}"
        )

        average_scores.pop("Micro_F1")
        average_scores.pop("Macro_F1")
        average_scores.pop("Micro_P")
        average_scores.pop("Macro_P")
        average_scores.pop("Micro_R")
        average_scores.pop("Macro_R")

    else:

        macro_average_measurement_p: float = 0.5 * (
            average_scores["MEASUREMENT"]["P"] + average_scores["QUAL_MEASUREMENT"]["P"]
        )
        macro_average_measurement_r: float = 0.5 * (
            average_scores["MEASUREMENT"]["R"] + average_scores["QUAL_MEASUREMENT"]["R"]
        )
        macro_average_measurement_f1: float = 0.5 * (
            average_scores["MEASUREMENT"]["F1"] + average_scores["QUAL_MEASUREMENT"]["F1"]
        )

        std_macro_average_measurement_p: float = round(
            100
            * np.std(
                [
                    0.5 * (rs["MEASUREMENT"]["P"] + rs["QUAL_MEASUREMENT"]["P"])
                    for rs in result_scores
                ]
            ),
            1,
        )
        std_macro_average_measurement_r: float = round(
            100
            * np.std(
                [
                    0.5 * (rs["MEASUREMENT"]["R"] + rs["QUAL_MEASUREMENT"]["R"])
                    for rs in result_scores
                ]
            ),
            1,
        )
        std_macro_average_measurement_f1: float = round(
            100
            * np.std(
                [
                    0.5 * (rs["MEASUREMENT"]["F"] + rs["QUAL_MEASUREMENT"]["F"])
                    for rs in result_scores
                ]
            ),
            1,
        )

        print(f"Average 2-Class Macro F1: {round(100 * macro_average_measurement_f1, 1)}")
        print(f"Std. Dev. of 2.Class Macro F1: {std_macro_average_measurement_f1}")
        print(
            f"Macro P: {round(100 * macro_average_measurement_p, 1)}; Std. Macro P: {std_macro_average_measurement_p}"
        )
        print(
            f"Macro R: {round(100 * macro_average_measurement_r, 1)}; Std. Macro R: {std_macro_average_measurement_r}"
        )

    for k in average_scores.keys():
        print(
            f"{k}: Precision: {round(100 * average_scores[k]['P'], 1)}, Std. P: {round(100 * average_scores[k]['P_std'], 1)}, Recall: {round(100 * average_scores[k]['R'], 1)}, Std. R: {round(100 * average_scores[k]['R_std'], 1)}, F1: {round(100 * average_scores[k]['F1'], 1)}, Std. F1: {round(100 * average_scores[k]['F1_std'], 1)}"
        )

    return


if __name__ == "__main__":
    main()

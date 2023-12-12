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
This function prints NER BILOU predictions in human readable format.
"""


def print_bilou_to_token_matchings(
    tokens: list[list[str]], predicted_tags: list[list[str]], gt_tags: list[list[str]]
) -> None:
    """
    Prints the predicted BILOU sequence against the true on onto the console and highlights the difference with color coding.

    Args:
        tokens (list[list[str]]): Batch of token lists
        predicted_tags (list[list[str]]): Batch of predicted sequences
        gt_tags (list[list[str]]): Batch of ground truth labels
    """
    for token_list, pred_list, gt_list in zip(tokens, predicted_tags, gt_tags):
        pred_string = "[ "
        gt_string = "[ "
        for p, g in zip(pred_list, gt_list):
            if p == g:
                pred_string += "\033[92m" + str(p) + ", " + "\033[0m"
                gt_string += "\033[92m" + str(g) + ", " + "\033[0m"
            else:
                pred_string += "\033[91m" + str(p) + ", " + "\033[0m"
                gt_string += "\033[91m" + str(g) + ", " + "\033[0m"
        pred_string += "]"
        gt_string += "]"
        print(f"Token: {token_list}")
        print(f"Pred: {pred_string}")
        print(f"GT: {gt_string}")

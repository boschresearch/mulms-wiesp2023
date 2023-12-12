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
This module contains metric compute functions for the NER task.
"""

from source.constants.mulms_constants import (
    mulms_ne_combined_dependency_labels,
    mulms_ne_labels,
)

B: str = "B-{0}"
I: str = "I-{0}"
L: str = "L-{0}"
O: str = "O"
U: str = "U-{0}"


def get_p_r_f_from_counts(label_counts: dict[str, dict]) -> dict:
    """
    Calculates precision, recall and F1 for predicted named entities given true positive (TP), false positive (FP), and false negative (FN) counts.

    Args:
        label_counts (dict[str, dict]): Dict containing TP, FP, and FN counts.

    Returns:
        dict: Micro- and Macro-F1 scores.
    """
    scores: dict = {"Micro_F1": 0.0, "Macro_F1": 0.0}
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    # Don't use combined labels for overall score calculation
    for label in set(label_counts.keys()).difference(mulms_ne_combined_dependency_labels):
        total_tp += label_counts[label]["TP"]
        total_fp += label_counts[label]["FP"]
        total_fn += label_counts[label]["FN"]
        try:
            scores[label] = {
                "P": label_counts[label]["TP"]
                / (label_counts[label]["TP"] + label_counts[label]["FP"]),
                "R": label_counts[label]["TP"]
                / (label_counts[label]["TP"] + label_counts[label]["FN"]),
            }
            scores[label]["F1"] = (
                2
                * scores[label]["P"]
                * scores[label]["R"]
                / (scores[label]["P"] + scores[label]["R"])
            )
            scores[label]["TP"] = label_counts[label]["TP"]
            scores[label]["FP"] = label_counts[label]["FP"]
            scores[label]["FN"] = label_counts[label]["FN"]
            scores["Macro_F1"] += scores[label]["F1"]
        except ZeroDivisionError:
            scores[label] = {"P": 0.0, "R": 0.0, "F1": 0.0}
            scores[label]["TP"] = label_counts[label]["TP"]
            scores[label]["FP"] = label_counts[label]["FP"]
            scores[label]["FN"] = label_counts[label]["FN"]
    scores["Macro_F1"] /= len(label_counts)
    try:
        micro_p: float = total_tp / (total_tp + total_fp)
        micro_r: float = total_tp / (total_tp + total_fn)
        scores["Micro_F1"] = 2 * micro_p * micro_r / (micro_p + micro_r)
    except ZeroDivisionError:
        scores["Micro_F1"] = 0.0
    return scores


def calculate_prec_recall_f1_for_dep_parsing(
    prediction: list[list[list[str]]], gold_labels: list[set[tuple]]
) -> dict:
    """
    Calculates P, R and F1 scores for dependency parsing based NER.

    Args:
        prediction (list[list[list[str]]]): Collection of 2D dependency matrices of shape (batch_size, max_sent_length, max_sent_length) which contain predictions
        gold_labels (list[set[tuple]]): Grount truth named entities of form (ENT, begin, end)

    Returns:
        dict: Micro F1, Macro F1 and P, R, F1 per label
    """
    label_counts: dict[str, dict] = dict(
        [(label, {"TP": 0, "FP": 0, "FN": 0, "count": 0}) for label in mulms_ne_labels]
    )
    set_of_predictions: list(set(tuple)) = []
    for pred in prediction:
        set_of_predictions.append(set())
        for i in range(len(pred)):
            for j in range(len(pred)):
                if pred[i][j] != "O":
                    if pred[i][j].count("+") > 0:  # Multi-Label
                        single_labels: list[str] = pred[i][j].split("+")
                        for sl in single_labels:
                            set_of_predictions[-1].add((sl, i, j))
                    else:
                        set_of_predictions[-1].add((pred[i][j], i, j))
    for pred_labels, gt_labels in zip(set_of_predictions, gold_labels):
        for gtl in gt_labels:
            label_counts[gtl[0]]["count"] += 1
        correctly_predicted_labels: set = gt_labels.intersection(pred_labels)
        wrongly_predicted_labels: set = pred_labels.difference(gt_labels)
        not_predicted_labels: set = gt_labels.difference(pred_labels)
        for cl in correctly_predicted_labels:
            label_counts[cl[0]]["TP"] += 1
        for wl in wrongly_predicted_labels:
            label_counts[wl[0]]["FP"] += 1
        for nl in not_predicted_labels:
            label_counts[nl[0]]["FN"] += 1

    return get_p_r_f_from_counts(label_counts)


def calculate_prec_recall_f1_for_bilou(
    predictions: list[list[str]],
    gold_labels: list[set[tuple]],
    ne_labels: list[str],
    return_predictions_per_sample: bool = False,
) -> dict:
    """
    Evaluates BILOU predictions produced by the sequence tagger model.

    Args:
        predictions (list[list[str]]): Batch of predicted NE labels for each token in each sentence
        gold_labels (list[set[tuple]]): Batch of tuples containing NE labels, begin token and end token
        ne_labels (list[str]): Underlying NE label set
        return_predictions_per_sample (bool, optional): Return predictions for each sample. Defaults to False.

    Returns:
        dict: Per label scores.
    """
    assert len(predictions) == len(
        gold_labels
    ), "Length of predictions and gold labels does not match."
    label_counts: dict[str, dict] = dict([(ne, {"TP": 0, "FP": 0, "FN": 0}) for ne in ne_labels])
    recombined_ne_predictions: list[set[tuple]] = []
    for pred_list in predictions:
        recombined_ne_predictions.append(set())
        stop_iterating: bool = False
        pred_list = [p.split("+") for p in pred_list]  # Splitting into single labels
        while not stop_iterating:
            stop_iterating = True
            for i in range(len(pred_list)):
                if len(pred_list[i]) > 0:
                    stop_iterating = False  # There are still predictions available
                    tag: str = pred_list[i][0].split("-")[-1]
                    if tag == O:
                        pred_list[i] = []  # Ignoring "O"
                    elif pred_list[i][0] == U.format(tag):  # Case of Unit length prediction
                        recombined_ne_predictions[-1].add(tuple([tag, i, i]))
                        pred_list[i] = pred_list[i][
                            1:
                        ]  # Remove first tag since it has been processed now
                    elif pred_list[i][0] == B.format(tag):  # Case of multi-token NEs
                        pred_list[i] = pred_list[i][1:]
                        is_valid_ne: bool = False
                        for j in range(
                            i + 1, i + 1 + len(pred_list[i + 1 :])
                        ):  # Look for all I and L labels in the following predictions
                            if len(pred_list[j]) == 0:  # Case is invalid; no following I or L tags
                                break
                            elif pred_list[j][0] == I.format(tag):  # Case is valid
                                pred_list[j] = pred_list[j][1:]
                            elif pred_list[j][0] == L.format(
                                tag
                            ):  # Case is valid, marks end of entity
                                is_valid_ne = True
                                pred_list[j] = pred_list[j][1:]
                                break
                            else:  # Case is invalid
                                break
                        if is_valid_ne:
                            recombined_ne_predictions[-1].add(tuple([tag, i, j]))
                        break  # Restart i-loop
                    else:  # Invalid annotation -> remove it from predictions and repeat iterating
                        pred_list[i] = pred_list[i][1:]

    # Now comparing predictions and gold labels
    for pred_labels, gt_labels in zip(recombined_ne_predictions, gold_labels):
        correctly_predicted_labels: set = gt_labels.intersection(pred_labels)
        wrongly_predicted_labels: set = pred_labels.difference(gt_labels)
        not_predicted_labels: set = gt_labels.difference(pred_labels)
        for cl in correctly_predicted_labels:
            label_counts[cl[0]]["TP"] += 1
        for wl in wrongly_predicted_labels:
            label_counts[wl[0]]["FP"] += 1
        for nl in not_predicted_labels:
            label_counts[nl[0]]["FN"] += 1

    if return_predictions_per_sample:
        return get_p_r_f_from_counts(label_counts), recombined_ne_predictions
    else:
        return get_p_r_f_from_counts(label_counts)


def calculate_prec_recall_f1_for_bio(
    predictions: list[list[str]],
    gold_labels: list[set[tuple]],
    ne_labels: list[str],
    return_predictions_per_sample: bool = False,
) -> dict:
    """
    Evaluates BIO (NOT BILOU!) predictions produced by the sequence tagger model.
    Args:
        predictions (list[list[str]]): Batch of predicted NE labels for each token in each sentence
        gold_labels (list[set[tuple]]): Batch of tuples containing NE labels, begin token and end token
        ne_labels (list[str]): Underlying NE label set
        return_predictions_per_sample (bool, optional): Return predictions for each sample. Defaults to False.
    Returns:
        dict: Per label scores.
    """
    assert len(predictions) == len(
        gold_labels
    ), "Length of predictions and gold labels does not match."
    label_counts: dict[str, dict] = dict([(ne, {"TP": 0, "FP": 0, "FN": 0}) for ne in ne_labels])
    recombined_ne_predictions: list[set[tuple]] = []
    for pred_list in predictions:
        recombined_ne_predictions.append(set())
        stop_iterating: bool = False
        pred_list = [p.split("+") for p in pred_list]  # Splitting into single labels
        while not stop_iterating:
            stop_iterating = True
            for i in range(len(pred_list)):
                if len(pred_list[i]) > 0:
                    stop_iterating = False  # There are still predictions available
                    tag: str = pred_list[i][0].split("-")[-1]
                    if tag == O:  # Ignoring "O"
                        pred_list[i] = []
                    elif pred_list[i][0] == B.format(tag):  # Starting B token
                        pred_list[i] = pred_list[i][1:]  # Remove it from the list
                        j: int = i + 1
                        while j < len(pred_list):  # Look for all I in the following predictions
                            if pred_list[j][0] == I.format(tag):  # Case is valid
                                pred_list[j] = pred_list[j][1:]
                            else:
                                break
                            j += 1
                        j -= 1  # Correct for one-step lookahead

                        recombined_ne_predictions[-1].add(tuple([tag, i, j]))
                        break  # Restart i-loop
                    else:  # Invalid annotation -> remove it from predictions and repeat iterating
                        pred_list[i] = pred_list[i][1:]
    # Now comparing predictions and gold labels
    for pred_labels, gt_labels in zip(recombined_ne_predictions, gold_labels):
        correctly_predicted_labels: set = gt_labels.intersection(pred_labels)
        wrongly_predicted_labels: set = pred_labels.difference(gt_labels)
        not_predicted_labels: set = gt_labels.difference(pred_labels)
        for cl in correctly_predicted_labels:
            label_counts[cl[0]]["TP"] += 1
        for wl in wrongly_predicted_labels:
            label_counts[wl[0]]["FP"] += 1
        for nl in not_predicted_labels:
            label_counts[nl[0]]["FN"] += 1
    if return_predictions_per_sample:
        return get_p_r_f_from_counts(label_counts), recombined_ne_predictions
    else:
        return get_p_r_f_from_counts(label_counts)

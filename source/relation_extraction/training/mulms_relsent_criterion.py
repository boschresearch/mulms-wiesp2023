#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from collections import defaultdict

import numpy as np

from source.relation_extraction.data_handling.mulms_relation_sentences import (
    MuLMSRelationSentence,
)


class MuLMSRelationSentenceValidationCriterion:
    def __init__(self, main_metric):
        self.main_metric = main_metric

        self.curr_epoch_counts = defaultdict(lambda: {"predicted": 0, "gold": 0, "correct": 0})
        self.metrics_format = None

        self.best_main_metric_value = None
        self.improvement = False

    def compute_and_log_pairwise_metrics(self, gold_relsent, predicted_relsent):
        assert isinstance(gold_relsent, MuLMSRelationSentence)
        assert isinstance(predicted_relsent, MuLMSRelationSentence)

        # Make sure that gold and prediction have the same basis
        assert gold_relsent.tokens == predicted_relsent.tokens
        assert len(gold_relsent.named_entities) == len(predicted_relsent.named_entities)
        for gold_ne, predicted_ne in zip(
            gold_relsent.named_entities, predicted_relsent.named_entities
        ):
            gold_start_ix, gold_end_ix = gold_ne[0]
            gold_ne_lbl = gold_ne[1]

            predicted_start_ix, predicted_end_ix = predicted_ne[0]
            predicted_ne_lbl = predicted_ne[1]

            assert gold_start_ix == predicted_start_ix
            assert gold_end_ix == predicted_end_ix
            assert gold_ne_lbl == predicted_ne_lbl

        num_ne = len(gold_relsent.named_entities)
        assert (
            len(gold_relsent.relation_matrix) == len(predicted_relsent.relation_matrix) == num_ne
        )
        assert all(len(row) == num_ne for row in gold_relsent.relation_matrix)
        assert all(len(row) == num_ne for row in predicted_relsent.relation_matrix)

        # Compute actual counts
        for i in range(len(gold_relsent.relation_matrix)):
            for j in range(len(gold_relsent.relation_matrix)):
                gold_rel = gold_relsent.relation_matrix[i][j]
                pred_rel = predicted_relsent.relation_matrix[i][j]

                if gold_rel != "[null]":
                    self.curr_epoch_counts[gold_rel]["gold"] += 1
                if pred_rel != "[null]":
                    self.curr_epoch_counts[pred_rel]["predicted"] += 1

                if gold_rel != "[null]" and pred_rel != "[null]":
                    if gold_rel == pred_rel:
                        self.curr_epoch_counts[gold_rel]["correct"] += 1

    def finalize_epoch(self, validation=False):
        curr_epoch_metrics = dict()

        # Compute per-label scores
        for rel_lbl in self.curr_epoch_counts:
            (
                curr_epoch_metrics[f"{rel_lbl}_gold"],
                curr_epoch_metrics[f"{rel_lbl}_predicted"],
                curr_epoch_metrics[f"{rel_lbl}_correct"],
            ) = (
                self.curr_epoch_counts[rel_lbl]["gold"],
                self.curr_epoch_counts[rel_lbl]["predicted"],
                self.curr_epoch_counts[rel_lbl]["correct"],
            )
            (
                curr_epoch_metrics[f"{rel_lbl}_precision"],
                curr_epoch_metrics[f"{rel_lbl}_recall"],
                curr_epoch_metrics[f"{rel_lbl}_f1"],
            ) = self._compute_prf(
                self.curr_epoch_counts[rel_lbl]["gold"],
                self.curr_epoch_counts[rel_lbl]["predicted"],
                self.curr_epoch_counts[rel_lbl]["correct"],
            )

        # Compute Macro scores
        curr_epoch_metrics["MACRO_precision"] = np.mean(
            [curr_epoch_metrics[f"{rel_lbl}_precision"] for rel_lbl in self.curr_epoch_counts]
        )
        curr_epoch_metrics["MACRO_recall"] = np.mean(
            [curr_epoch_metrics[f"{rel_lbl}_recall"] for rel_lbl in self.curr_epoch_counts]
        )
        curr_epoch_metrics["MACRO_f1"] = np.mean(
            [curr_epoch_metrics[f"{rel_lbl}_f1"] for rel_lbl in self.curr_epoch_counts]
        )

        # Compute Micro scores
        curr_epoch_metrics["OVERALL_gold"] = sum(
            self.curr_epoch_counts[rel_lbl]["gold"] for rel_lbl in self.curr_epoch_counts
        )
        curr_epoch_metrics["OVERALL_predicted"] = sum(
            self.curr_epoch_counts[rel_lbl]["predicted"] for rel_lbl in self.curr_epoch_counts
        )
        curr_epoch_metrics["OVERALL_correct"] = sum(
            self.curr_epoch_counts[rel_lbl]["correct"] for rel_lbl in self.curr_epoch_counts
        )
        (
            curr_epoch_metrics["MICRO_precision"],
            curr_epoch_metrics["MICRO_recall"],
            curr_epoch_metrics["MICRO_f1"],
        ) = self._compute_prf(
            curr_epoch_metrics["OVERALL_gold"],
            curr_epoch_metrics["OVERALL_predicted"],
            curr_epoch_metrics["OVERALL_correct"],
        )

        if validation:
            if self.main_metric in curr_epoch_metrics:
                curr_main_metric_value = curr_epoch_metrics[self.main_metric]
            else:
                curr_main_metric_value = 0.0
            if (
                self.best_main_metric_value is None
                or curr_main_metric_value > self.best_main_metric_value
            ):
                self.best_main_metric_value = curr_main_metric_value
                self.improvement = True
            else:
                self.improvement = False

        self.curr_epoch_counts = defaultdict(lambda: {"predicted": 0, "gold": 0, "correct": 0})

        return curr_epoch_metrics

    def _compute_prf(self, num_gold, num_predicted, num_correct):
        precision = num_correct / num_predicted if num_predicted else 0.0
        recall = num_correct / num_gold if num_gold else 0.0
        fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return precision, recall, fscore

    def last_epoch_improved_best(self):
        return self.improvement

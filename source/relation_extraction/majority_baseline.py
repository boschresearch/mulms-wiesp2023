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

from collections import namedtuple
from copy import deepcopy
from os.path import join

from source.constants.mulms_constants import MULMS_PATH, PROJECT_ROOT
from source.relation_extraction.data_handling.mulms_rel_dataloader import (
    get_relation_dataloader,
)
from source.relation_extraction.data_handling.mulms_relation_sentences import (
    MuLMSRelationSentence,
)
from source.relation_extraction.training.mulms_relsent_criterion import (
    MuLMSRelationSentenceValidationCriterion,
)
from source.relation_extraction.vocab import BasicVocab

TOP_RELATION_BY_NE_TYPE_PAIR = {
    "MAT": {
        "MAT": "[null]",
        "DEV": "[null]",
        "FORM": "[null]",
        "PROPERTY": "[null]",
        "MEASUREMENT": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "TECHNIQUE": "[null]",
        "CITE": "[null]",
        "INSTRUMENT": "[null]",
        "SAMPLE": "[null]",
    },
    "DEV": {
        "MAT": "[null]",
        "DEV": "[null]",
        "PROPERTY": "[null]",
        "MEASUREMENT": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "FORM": "[null]",
        "CITE": "[null]",
        "INSTRUMENT": "[null]",
        "SAMPLE": "[null]",
        "TECHNIQUE": "[null]",
    },
    "FORM": {
        "MAT": "[null]",
        "FORM": "[null]",
        "PROPERTY": "[null]",
        "MEASUREMENT": "[null]",
        "DEV": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "TECHNIQUE": "[null]",
        "CITE": "[null]",
        "INSTRUMENT": "[null]",
        "SAMPLE": "[null]",
    },
    "PROPERTY": {
        "PROPERTY": "[null]",
        "MAT": "[null]",
        "MEASUREMENT": "[null]",
        "DEV": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "FORM": "[null]",
        "TECHNIQUE": "[null]",
        "INSTRUMENT": "[null]",
        "CITE": "[null]",
        "SAMPLE": "[null]",
    },
    "MEASUREMENT": {
        "PROPERTY": "measuresProperty",
        "MAT": "[null]",
        "MEASUREMENT": "[null]",
        "DEV": "[null]",
        "VALUE": "conditionPropertyValue",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "FORM": "[null]",
        "TECHNIQUE": "usesTechnique",
        "CITE": "takenFrom",
        "INSTRUMENT": "conditionInstrument",
        "SAMPLE": "conditionSampleFeatures",
    },
    "VALUE": {
        "PROPERTY": "[null]",
        "MAT": "[null]",
        "MEASUREMENT": "[null]",
        "DEV": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "FORM": "[null]",
        "CITE": "[null]",
        "INSTRUMENT": "[null]",
        "TECHNIQUE": "[null]",
        "SAMPLE": "[null]",
    },
    "RANGE": {
        "PROPERTY": "[null]",
        "MAT": "[null]",
        "MEASUREMENT": "[null]",
        "DEV": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "FORM": "[null]",
        "CITE": "[null]",
        "INSTRUMENT": "[null]",
        "TECHNIQUE": "[null]",
        "SAMPLE": "[null]",
    },
    "NUM": {
        "PROPERTY": "[null]",
        "MAT": "[null]",
        "MEASUREMENT": "[null]",
        "DEV": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "FORM": "[null]",
        "CITE": "[null]",
        "INSTRUMENT": "[null]",
        "TECHNIQUE": "[null]",
        "SAMPLE": "[null]",
    },
    "UNIT": {
        "PROPERTY": "[null]",
        "MAT": "[null]",
        "MEASUREMENT": "[null]",
        "DEV": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "FORM": "[null]",
        "CITE": "[null]",
        "INSTRUMENT": "[null]",
        "TECHNIQUE": "[null]",
        "SAMPLE": "[null]",
    },
    "TECHNIQUE": {
        "PROPERTY": "[null]",
        "TECHNIQUE": "[null]",
        "MAT": "[null]",
        "MEASUREMENT": "[null]",
        "FORM": "[null]",
        "INSTRUMENT": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "SAMPLE": "[null]",
        "CITE": "[null]",
        "DEV": "[null]",
    },
    "CITE": {
        "MAT": "[null]",
        "DEV": "[null]",
        "CITE": "[null]",
        "NUM": "[null]",
        "VALUE": "[null]",
        "UNIT": "[null]",
        "RANGE": "[null]",
        "FORM": "[null]",
        "MEASUREMENT": "[null]",
        "SAMPLE": "[null]",
        "PROPERTY": "[null]",
        "TECHNIQUE": "[null]",
        "INSTRUMENT": "[null]",
    },
    "INSTRUMENT": {
        "PROPERTY": "[null]",
        "FORM": "[null]",
        "MEASUREMENT": "[null]",
        "INSTRUMENT": "[null]",
        "TECHNIQUE": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "UNIT": "[null]",
        "MAT": "[null]",
        "DEV": "[null]",
        "CITE": "[null]",
        "SAMPLE": "[null]",
    },
    "SAMPLE": {
        "SAMPLE": "[null]",
        "CITE": "[null]",
        "UNIT": "[null]",
        "MAT": "[null]",
        "FORM": "[null]",
        "VALUE": "[null]",
        "RANGE": "[null]",
        "NUM": "[null]",
        "PROPERTY": "[null]",
        "MEASUREMENT": "[null]",
        "TECHNIQUE": "[null]",
        "DEV": "[null]",
        "INSTRUMENT": "[null]",
    },
}


def put_majority_rels(rel_sentence: MuLMSRelationSentence):
    """
    Computes the most frequently occuring relation type per label pair.

    Args:
        rel_sentence (MuLMSRelationSentence): A relation sentence with relation matrices.
    """
    for i, head_ne in enumerate(rel_sentence.named_entities):
        for j, dep_ne in enumerate(rel_sentence.named_entities):
            head_ne_type = head_ne[1]
            dep_ne_type = dep_ne[1]

            maj_rel = TOP_RELATION_BY_NE_TYPE_PAIR[head_ne_type][dep_ne_type]
            rel_sentence.relation_matrix[i][j] = maj_rel


def compute_majority_baseline_metrics(split: str = "test") -> dict:
    """
    Computes the scores when using the majority baseline for relation extraction.
    For each pair of entities, the majority baseline always predicts the most occurring
    relation between these two entities.

    Args:
        split (str, optional): Desired MuLMS split. Defaults to "test".

    Returns:
        Dict: Dict containing all average and per-label scores
    """
    DummyModel = namedtuple("DummyModel", "factorized rel_vocab ne_vocab")
    dummy_model = DummyModel(
        factorized=False,
        rel_vocab=BasicVocab(
            vocab_filename=join(PROJECT_ROOT, "source/relation_extraction/vocabs/rel.vocab")
        ),
        ne_vocab=BasicVocab(
            vocab_filename=join(PROJECT_ROOT, "source/relation_extraction/vocabs/ne.vocab")
        ),
    )

    if split == "dev":
        data_loader = get_relation_dataloader(MULMS_PATH, "validation", dummy_model, 32)
    elif split == "test":
        data_loader = get_relation_dataloader(MULMS_PATH, "test", dummy_model, 32)
    else:
        raise

    eval_criterion = MuLMSRelationSentenceValidationCriterion("MICRO_f1")

    for batch in data_loader:
        _, gold_relsents, _ = batch
        for gold_relsent in gold_relsents:
            predicted_relsent = deepcopy(gold_relsent)
            put_majority_rels(predicted_relsent)

            eval_criterion.compute_and_log_pairwise_metrics(gold_relsent, predicted_relsent)

    return eval_criterion.finalize_epoch(validation=True)


if __name__ == "__main__":
    maj_baseline_metrics_eval = compute_majority_baseline_metrics(split="test")
    maj_baseline_metrics_dev = compute_majority_baseline_metrics(split="dev")

    assert maj_baseline_metrics_eval.keys() == maj_baseline_metrics_dev.keys()

    avg_maj_baseline_metrics: dict = dict()
    std_maj_baseline_metrics: dict = dict()
    for metric_name in maj_baseline_metrics_eval.keys():
        avg_maj_baseline_metrics[metric_name + "_eval"] = maj_baseline_metrics_eval[metric_name]
        avg_maj_baseline_metrics[metric_name + "_dev"] = maj_baseline_metrics_dev[metric_name]

        std_maj_baseline_metrics[metric_name + "_eval"] = 0.0
        std_maj_baseline_metrics[metric_name + "_dev"] = 0.0

    print("Results:")
    for k, v in avg_maj_baseline_metrics.items():
        print(k, v)

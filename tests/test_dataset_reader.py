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
This module contains some tests for the MuLMS dataset reader.
"""

import pandas as pd
from datasets import Split, load_dataset

from source.constants.mulms_constants import (
    MULMS_DATASET_READER_PATH,
    MULMS_FILES,
    MULMS_PATH,
    SOFC_DATASET_READER_PATH,
    SOFC_PATH,
)


class TestDatasetReader:
    """
    The test class.
    """

    def __init__(self) -> None:
        """
        Initializes this class by defining to class members.
        """
        self._mulms_dataset = None
        self._sofc_dataset = None

    def test_reading(self, corpus_name: str):
        """
        Checks if the MuLMS and SOFC datasets are readable and loadable.

        Args:
            corpus_name (str): The subpart of MuLMS.
        """
        self._mulms_dataset = load_dataset(
            MULMS_DATASET_READER_PATH.__str__(),
            data_dir=MULMS_PATH.__str__(),
            data_files=MULMS_FILES,
            name=corpus_name,
        )
        self._sofc_dataset = load_dataset(
            SOFC_DATASET_READER_PATH.__str__(), data_dir=SOFC_PATH.__str__()
        )

    def create_dataframes(self) -> tuple:
        """
        Use this method to get pandas Dataframes which allow you to easily inspect the dataset.

        Returns:
            tuple: Tuple of Dataframes; one DF for each split
        """
        if self._mulms_dataset is None:
            self.test_reading("MuLMS_Corpus")
        df_train_mulms_corpus: pd.DataFrame = pd.DataFrame(
            {
                "doc_id": self._mulms_dataset[Split.TRAIN]["doc_id"],
                "sentence": self._mulms_dataset[Split.TRAIN]["sentence"],
                "tokens": self._mulms_dataset[Split.TRAIN]["tokens"],
                "beginOffset": self._mulms_dataset[Split.TRAIN]["beginOffset"],
                "endOffset": self._mulms_dataset[Split.TRAIN]["endOffset"],
                "AZ_labels": self._mulms_dataset[Split.TRAIN]["AZ_labels"],
                "Measurement_label": self._mulms_dataset[Split.TRAIN]["Measurement_label"],
                "NER_labels": self._mulms_dataset[Split.TRAIN]["NER_labels"],
                "NER_labels_BILOU": self._mulms_dataset[Split.TRAIN]["NER_labels_BILOU"],
                "relations": self._mulms_dataset[Split.TRAIN]["relations"],
                "docFileName": self._mulms_dataset[Split.TRAIN]["docFileName"],
                "data_split": self._mulms_dataset[Split.TRAIN]["data_split"],
                "category_info": self._mulms_dataset[Split.TRAIN]["category"],
            }
        )
        df_dev_mulms_corpus: pd.DataFrame = pd.DataFrame(
            {
                "doc_id": self._mulms_dataset[Split.VALIDATION]["doc_id"],
                "sentence": self._mulms_dataset[Split.VALIDATION]["sentence"],
                "tokens": self._mulms_dataset[Split.VALIDATION]["tokens"],
                "beginOffset": self._mulms_dataset[Split.VALIDATION]["beginOffset"],
                "endOffset": self._mulms_dataset[Split.VALIDATION]["endOffset"],
                "AZ_labels": self._mulms_dataset[Split.VALIDATION]["AZ_labels"],
                "Measurement_label": self._mulms_dataset[Split.VALIDATION]["Measurement_label"],
                "NER_labels": self._mulms_dataset[Split.VALIDATION]["NER_labels"],
                "NER_labels_BILOU": self._mulms_dataset[Split.VALIDATION]["NER_labels_BILOU"],
                "relations": self._mulms_dataset[Split.VALIDATION]["relations"],
                "docFileName": self._mulms_dataset[Split.VALIDATION]["docFileName"],
                "data_split": self._mulms_dataset[Split.VALIDATION]["data_split"],
                "category_info": self._mulms_dataset[Split.VALIDATION]["category"],
            }
        )
        df_test_mulms_corpus: pd.DataFrame = pd.DataFrame(
            {
                "doc_id": self._mulms_dataset[Split.TEST]["doc_id"],
                "sentence": self._mulms_dataset[Split.TEST]["sentence"],
                "tokens": self._mulms_dataset[Split.TEST]["tokens"],
                "beginOffset": self._mulms_dataset[Split.TEST]["beginOffset"],
                "endOffset": self._mulms_dataset[Split.TEST]["endOffset"],
                "AZ_labels": self._mulms_dataset[Split.TEST]["AZ_labels"],
                "Measurement_label": self._mulms_dataset[Split.TEST]["Measurement_label"],
                "NER_labels": self._mulms_dataset[Split.TEST]["NER_labels"],
                "NER_labels_BILOU": self._mulms_dataset[Split.TEST]["NER_labels_BILOU"],
                "relations": self._mulms_dataset[Split.TEST]["relations"],
                "docFileName": self._mulms_dataset[Split.TEST]["docFileName"],
                "data_split": self._mulms_dataset[Split.TEST]["data_split"],
                "category_info": self._mulms_dataset[Split.TEST]["category"],
            }
        )

        df_train_sofc_corpus: pd.DataFrame = pd.DataFrame(
            {
                "text": self._sofc_dataset[Split.TRAIN]["text"],
                "sentence_offsets": self._sofc_dataset[Split.TRAIN]["sentence_offsets"],
                "sentences": self._sofc_dataset[Split.TRAIN]["sentences"],
                "sentence_labels": self._sofc_dataset[Split.TRAIN]["sentence_labels"],
                "token_offsets": self._sofc_dataset[Split.TRAIN]["token_offsets"],
                "tokens": self._sofc_dataset[Split.TRAIN]["tokens"],
                "entity_labels": self._sofc_dataset[Split.TRAIN]["entity_labels"],
                "slot_labels": self._sofc_dataset[Split.TRAIN]["slot_labels"],
                "links": self._sofc_dataset[Split.TRAIN]["links"],
                "slots": self._sofc_dataset[Split.TRAIN]["slots"],
                "spans": self._sofc_dataset[Split.TRAIN]["spans"],
                "experiments": self._sofc_dataset[Split.TRAIN]["experiments"],
            }
        )

        df_dev_sofc_corpus: pd.DataFrame = pd.DataFrame(
            {
                "text": self._sofc_dataset[Split.VALIDATION]["text"],
                "sentence_offsets": self._sofc_dataset[Split.VALIDATION]["sentence_offsets"],
                "sentences": self._sofc_dataset[Split.VALIDATION]["sentences"],
                "sentence_labels": self._sofc_dataset[Split.VALIDATION]["sentence_labels"],
                "token_offsets": self._sofc_dataset[Split.VALIDATION]["token_offsets"],
                "tokens": self._sofc_dataset[Split.VALIDATION]["tokens"],
                "entity_labels": self._sofc_dataset[Split.VALIDATION]["entity_labels"],
                "slot_labels": self._sofc_dataset[Split.VALIDATION]["slot_labels"],
                "links": self._sofc_dataset[Split.VALIDATION]["links"],
                "slots": self._sofc_dataset[Split.VALIDATION]["slots"],
                "spans": self._sofc_dataset[Split.VALIDATION]["spans"],
                "experiments": self._sofc_dataset[Split.VALIDATION]["experiments"],
            }
        )

        df_test_sofc_corpus: pd.DataFrame = pd.DataFrame(
            {
                "text": self._sofc_dataset[Split.TEST]["text"],
                "sentence_offsets": self._sofc_dataset[Split.TEST]["sentence_offsets"],
                "sentences": self._sofc_dataset[Split.TEST]["sentences"],
                "sentence_labels": self._sofc_dataset[Split.TEST]["sentence_labels"],
                "token_offsets": self._sofc_dataset[Split.TEST]["token_offsets"],
                "tokens": self._sofc_dataset[Split.TEST]["tokens"],
                "entity_labels": self._sofc_dataset[Split.TEST]["entity_labels"],
                "slot_labels": self._sofc_dataset[Split.TEST]["slot_labels"],
                "links": self._sofc_dataset[Split.TEST]["links"],
                "slots": self._sofc_dataset[Split.TEST]["slots"],
                "spans": self._sofc_dataset[Split.TEST]["spans"],
                "experiments": self._sofc_dataset[Split.TEST]["experiments"],
            }
        )

        self.test_reading("NER_Dependencies")

        df_train_ne_dependencies: pd.DataFrame = pd.DataFrame(
            {
                "ID": self._mulms_dataset[Split.TRAIN]["ID"],
                "Sentence": self._mulms_dataset[Split.TRAIN]["sentence"],
                "Token_ID": self._mulms_dataset[Split.TRAIN]["token_id"],
                "Token_Text": self._mulms_dataset[Split.TRAIN]["token_text"],
                "NE_Dependencies": self._mulms_dataset[Split.TRAIN]["NE_Dependencies"],
                "data_split": self._mulms_dataset[Split.TRAIN]["data_split"],
            }
        )

        df_dev_ne_dependencies: pd.DataFrame = pd.DataFrame(
            {
                "ID": self._mulms_dataset[Split.VALIDATION]["ID"],
                "Sentence": self._mulms_dataset[Split.VALIDATION]["sentence"],
                "Token_ID": self._mulms_dataset[Split.VALIDATION]["token_id"],
                "Token_Text": self._mulms_dataset[Split.VALIDATION]["token_text"],
                "NE_Dependencies": self._mulms_dataset[Split.VALIDATION]["NE_Dependencies"],
                "data_split": self._mulms_dataset[Split.VALIDATION]["data_split"],
            }
        )

        df_test_ne_dependencies: pd.DataFrame = pd.DataFrame(
            {
                "ID": self._mulms_dataset[Split.TEST]["ID"],
                "Sentence": self._mulms_dataset[Split.TEST]["sentence"],
                "Token_ID": self._mulms_dataset[Split.TEST]["token_id"],
                "Token_Text": self._mulms_dataset[Split.TEST]["token_text"],
                "NE_Dependencies": self._mulms_dataset[Split.TEST]["NE_Dependencies"],
                "data_split": self._mulms_dataset[Split.TEST]["data_split"],
            }
        )
        # You can use this return as a breakpoint for your debugger too -> you can now inspect the dataset
        return (
            df_train_mulms_corpus,
            df_dev_mulms_corpus,
            df_test_mulms_corpus,
            df_train_ne_dependencies,
            df_dev_ne_dependencies,
            df_test_ne_dependencies,
            df_train_sofc_corpus,
            df_dev_sofc_corpus,
            df_test_sofc_corpus,
        )


tdr = TestDatasetReader()
tdr.create_dataframes()

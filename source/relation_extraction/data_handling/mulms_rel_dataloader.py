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

from torch.utils.data import DataLoader

from source.relation_extraction.data_handling.mulms_relation_sentences import (
    MuLMSRelationSentence,
)
from source.utils.helpers import load_mulms_hf_dataset


def get_relation_dataloader(
    corpus_path,
    split_name,
    model,
    batch_size,
    shuffle=True,
    num_workers=1,
    fold=None,
    reverse_fold=False,
):

    dataset = load_mulms_hf_dataset(split_name)

    if fold is not None:
        if not reverse_fold:
            allowed_folds = {split_name + str(fold)}
        else:
            allowed_folds = {split_name + str(i) for i in range(1, 6)} - {split_name + str(fold)}
        converted_dataset = [
            MuLMSRelationSentence(sent["tokens"], sent["NER_labels"], sent["relations"])
            for sent in dataset
            if sent["data_split"] in allowed_folds
        ]
    else:
        converted_dataset = [
            MuLMSRelationSentence(sent["tokens"], sent["NER_labels"], sent["relations"])
            for sent in dataset
        ]

    return DataLoader(
        converted_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: MuLMSRelationSentence.batchify(x, model, factorized=model.factorized),
    )
